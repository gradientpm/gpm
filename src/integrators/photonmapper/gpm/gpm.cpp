/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <mitsuba/core/plugin.h>
#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>

#include <mitsuba/bidir/manifold.h>

#if defined(MTS_OPENMP)
# include <omp.h>
#include <mitsuba/bidir/mempool.h>
#endif

#include "gpm_proc.h"
#include "gpm_struct.h"

#include "gpm_gatherpoint.h"

// Poisson solver
// Same as used in GPT
#include "../../poisson_solver/Solver.hpp"

MTS_NAMESPACE_BEGIN

/*!\plugin{sppm}{Stochastic progressive photon mapping integrator}
 * \order{8}
 * \parameters{
 *     \parameter{maxDepth}{\Integer}{Specifies the longest path depth
 *         in the generated output image (where \code{-1} corresponds to $\infty$).
 *	       A value of \code{1} will only render directly visible light sources.
 *	       \code{2} will lead to single-bounce (direct-only) illumination,
 *	       and so on. \default{\code{-1}}
 *	   }
 *     \parameter{photonCount}{\Integer}{Number of photons to be shot per iteration\default{250000}}
 *     \parameter{initialRadius}{\Float}{Initial radius of gather points in world space units.
 *         \default{0, i.e. decide automatically}}
 *     \parameter{alpha}{\Float}{Radius reduction parameter \code{alpha} from the paper\default{0.7}}
 *     \parameter{granularity}{\Integer}{
		Granularity of photon tracing work units for the purpose
		of parallelization (in \# of shot particles) \default{0, i.e. decide automatically}
 *     }
 *	   \parameter{rrDepth}{\Integer}{Specifies the minimum path depth, after
 *	      which the implementation will start to use the ``russian roulette''
 *	      path termination criterion. \default{\code{5}}
 *	   }
 *     \parameter{maxPasses}{\Integer}{Maximum number of passes to render (where \code{-1}
 *        corresponds to rendering until stopped manually). \default{\code{-1}}}
 * }
 * This plugin implements stochastic progressive photon mapping by Hachisuka et al.
 * \cite{Hachisuka2009Stochastic}. This algorithm is an extension of progressive photon
 * mapping (\pluginref{ppm}) that improves convergence
 * when rendering scenes involving depth-of-field, motion blur, and glossy reflections.
 *
 * Note that the implementation of \pluginref{sppm} in Mitsuba ignores the sampler
 * configuration---hence, the usual steps of choosing a sample generator and a desired
 * number of samples per pixel are not necessary. As with \pluginref{ppm}, once started,
 * the rendering process continues indefinitely until it is manually stopped.
 *
 * \remarks{
 *    \item Due to the data dependencies of this algorithm, the parallelization is
 *    limited to the local machine (i.e. cluster-wide renderings are not implemented)
 *    \item This integrator does not handle participating media
 *    \item This integrator does not currently work with subsurface scattering
 *    models.
 * }
 */

// Initialization of the static value
Float VertexClassifier::roughnessThreshold = 0.1f;

class GPMIntegrator : public Integrator {
public:
  	GPMIntegrator(const Properties &props) : Integrator(props) {
        m_config.load(props);

		// Generate the gather point in the similar way for all PPM/SPPM codes
		m_mutex = new Mutex();
		m_gpManager = new RadiusInitializer(m_config);
		VertexClassifier::roughnessThreshold = m_config.bounceRoughness;
	}

  	GPMIntegrator(Stream *stream, InstanceManager *manager)
	 : Integrator(stream, manager) { }

	void serialize(Stream *stream, InstanceManager *manager) const {
		Integrator::serialize(stream, manager);
		Log(EError, "Network rendering is not supported!");
	}

	void cancel() {
		m_running = false;
	}


	bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
			int sceneResID, int sensorResID, int samplerResID) {
		Integrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);


		if(m_config.fluxLightDensity)
			const_cast<Scene*>(scene)->weightEmitterFlux();

		if (m_config.initialRadius == 0) {
			/* Guess an initial radius if not provided
			  (use scene width / horizontal or vertical pixel count) * 5 */
			Float rad = scene->getBSphere().radius;
			Vector2i filmSize = scene->getSensor()->getFilm()->getSize();

            m_config.initialRadius = std::min(rad / filmSize.x, rad / filmSize.y) * 5;
		}

#if 1
        ///////////////////////////
        // Check scene requirement

		// 1) Check if all the lights are Area ones
		auto emitters = scene->getEmitters();
		for(auto e: emitters) {
			std::string emitterType = e->getClass()->getName();
			SLog(EInfo, "Emitter type: %s", emitterType.c_str());
			if(emitterType != "AreaLight") {
				SLog(EError, "Unsupported light type: %s", emitterType.c_str());
			}
		}

        bool errorInBSDF = false;
		// 2) Check all BSDF:
		auto shapes = scene->getShapes();
		for(auto s: shapes) {
			const BSDF* bsdf = s->getBSDF();
			if(bsdf == 0 || bsdf->getComponentCount() == 0) continue;

			// We need to mimic an intersection information
			// Just sample the shape
			PositionSamplingRecord pRec(0.f);
			s->samplePosition(pRec,Point2(0.5f));
			Ray ray(pRec.p + pRec.n*0.0001, pRec.n, 0.f);
			Intersection its;
			if(!scene->rayIntersect(ray, its)) {
				SLog(EWarn, "Intersection error for query the bsdf shape information?!");
				continue;
			}

            // Early exception if pdfComponent is not implemented
            // It just to be sure that everything is Ok in case
            // of non supported BSDF
            {
                BSDFSamplingRecord bRec(its, Vector());
                bRec.component = 0;
                bsdf->pdfComponent(bRec);
            }
		}

        if(errorInBSDF) {
          SLog(EError, "Quit");
        }
#endif

		return true;
	}

  bool render(Scene *scene, RenderQueue *queue,
			  const RenderJob *job, int sceneResID, int sensorResID, int unused) {
	  ref<Scheduler> sched = Scheduler::getInstance();
	  ref<Sensor> sensor = scene->getSensor();
	  ref<Film> film = sensor->getFilm();
	  size_t nCores = sched->getCoreCount();
	  Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
		  film->getCropSize().x, film->getCropSize().y,
		  nCores, nCores == 1 ? "core" : "cores");

	  Vector2i cropSize = film->getCropSize();
	  Point2i cropOffset = film->getCropOffset();

	  m_gatherBlocks.clear();
	  m_running = true;
	  m_totalEmitted = 0;
	  m_totalPhotons = 0;

	  ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
			  createObject(MTS_CLASS(Sampler), Properties("independent")));
	  int blockSize = scene->getBlockSize();

      /* Create CSV file to dump all the rendering timings */
      // Also create an timer
      std::string timeFilename = scene->getDestinationFile().string()
          + "_time.csv";
      std::ofstream timeFile(timeFilename.c_str());
      ref<Timer> renderingTimer = new Timer;

      // Allocate throughput image
      m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
      m_bitmap->clear();

      // Allocate gradient images
      m_bitmapGx = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
      m_bitmapGx->clear();
      m_bitmapGy = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
      m_bitmapGy->clear();

      // Allocate reconstruction
      m_bitmapRecons = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
      m_bitmapRecons->clear();

      // Allocate gather points for each image block
      for (int yofs = 0; yofs < cropSize.y; yofs += blockSize) {
		  for (int xofs = 0; xofs < cropSize.x; xofs += blockSize) {
			  m_gatherBlocks.push_back(std::vector<GatherPoint>());
			  m_offset.push_back(Point2i(cropOffset.x + xofs, cropOffset.y + yofs));
			  std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[m_gatherBlocks.size() - 1];
			  int nPixels = std::min(blockSize, cropSize.y - yofs)
					      * std::min(blockSize, cropSize.x - xofs);
			  gatherPoints.resize(nPixels);

			  for (int i = 0; i < nPixels; ++i) {
                  gatherPoints[i].radius = m_config.initialRadius;
                  gatherPoints[i].scale = m_config.initialScale;
			  }
		  }
	  }

	  // Initialize the class responsible to the GP generation
	  // and the radii initialization
      m_gpManager->init(scene, m_config.maxDepth, m_config.rrDepth,
                        m_gatherBlocks, m_offset);

	  // Initialize memory for thread objects
	  m_threadData.resize(nCores);
	  for(int i = 0; i < nCores; i++) {
		  m_threadData[i] = new GPMThreadData(scene, m_config);
	  }

	  /* Create a sampler instance for every core */
	  /* and a memory pool */
	  std::vector<SerializableObject *> samplers(sched->getCoreCount());
	  std::vector<MemoryPool> memoryPools(sched->getCoreCount());
	  for (size_t i=0; i<sched->getCoreCount(); ++i) {
		  ref<Sampler> clonedSampler = sampler->clone();
		  clonedSampler->incRef();
		  samplers[i] = clonedSampler.get();
	  }

	  int samplerResID = sched->registerMultiResource(samplers);

#ifdef MTS_DEBUG_FP
	  enableFPExceptions();
#endif

#if defined(MTS_OPENMP)
	  Thread::initializeOpenMP(nCores);
#endif

	  int it = 0;
      while (m_running && (m_config.maxPasses == -1 || it < m_config.maxPasses)) {
          // Step 1: Generate the gather points and scale their flux
          Log(EInfo, "Pass %d / %d", it + 1, m_config.maxPasses);
		  Log(EInfo, "Regenerating gather points positions and radius!");
		  m_gpManager->regeneratePositionAndRadius();
		  Log(EInfo, "Done regenerating!");
		  m_gpManager->rescaleFlux();

		  if(it == 0) {
			  /* Create a copy of the gather point using image space,
               * to be able to retrieve them easily when we compute gradients for example.
               */
			  m_imgGP.resize(cropSize.x);
			  for (int x = 0; x < cropSize.x; x += 1) {
				  m_imgGP[x].resize(cropSize.y);
			  }
			  for(int idBlock = 0; idBlock < m_gatherBlocks.size(); idBlock++) {
				  std::vector<GatherPoint>& currBlock = m_gatherBlocks[idBlock];
				  for(GatherPoint& gp: currBlock) {
					  m_imgGP[gp.pos.x][gp.pos.y] = &gp;
				  }
			  }
		  }

          // Step 2: Shoot the photon, collect them, compute gradient and reconstruct the image
          photonMapPass(scene, ++it, queue, job, film, sceneResID,
						sensorResID, samplerResID,
						memoryPools);
          
          // This code is more to display some statistics
          unsigned int milliseconds = renderingTimer->getMilliseconds();
          timeFile << (milliseconds / 1000.f) << ",\n";
          timeFile.flush();
          Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
              milliseconds % 1000);
          renderingTimer->reset();
          Statistics::getInstance()->printStats();
	  }

      // Before stopping, display the results
      if (!m_config.showOnlyRecons) {
		  film->setBitmap(m_bitmap);
		  queue->signalRefresh(job);
      } else {
          film->setBitmap(m_bitmapRecons);
          queue->signalRefresh(job);
      }

#ifdef MTS_DEBUG_FP
	  disableFPExceptions();
#endif

      // Delete allocated objects
	  for (size_t i=0; i<samplers.size(); ++i)
		  samplers[i]->decRef();

      timeFile.close();
	  sched->unregisterResource(samplerResID);
	  return true;
  }

  void photonMapPass(Scene* scene, int it,
					 RenderQueue *queue, const RenderJob *job,
					 Film *film, int sceneResID, int sensorResID, int samplerResID,
					 std::vector<MemoryPool>& memoryPools) {
	  Log(EInfo, "Performing a photon mapping pass %i (" SIZE_T_FMT " photons so far)",
		  it, m_totalPhotons);
	  ref<Scheduler> sched = Scheduler::getInstance();

	  /* Re-get the image size useful for testing if
       * a gather point is at the image border */
	  Vector2i cropSize = film->getCropSize();

	  /* Generate the global photon map */
      ref<GradientPhotonProcess> proc = new GradientPhotonProcess(m_config.photonCount,
                                                                  m_config.granularity,
																  m_config.maxDepth, m_config.rrDepth, true,
                                                                  m_config.autoCancelGathering, job, memoryPools,
                                                                  m_config.minDepth);
	  proc->bindResource("scene", sceneResID);
	  proc->bindResource("sensor", sensorResID);
	  proc->bindResource("sampler", samplerResID);
	  sched->schedule(proc);
	  sched->wait(proc);

      // After the photon generated, we build the photon map.
	  ref<GPhotonMap> photonMap = proc->getPhotonMap();
	  photonMap->build();
	  Log(EDebug, "Photon map full. Shot " SIZE_T_FMT " paths, excess paths due to parallelism: "
			  SIZE_T_FMT, proc->getShotParticles(), proc->getExcessPhotons());

	  Log(EInfo, "Gathering ..");
	  m_totalEmitted += proc->getShotParticles();
	  m_totalPhotons += photonMap->size();
	  film->clear();

      // Collect the photon by iterating over the gather point
#if defined(MTS_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
	  for (int blockIdx = 0; blockIdx < (int)m_gatherBlocks.size(); ++blockIdx) {
		  // Get specific thread information (samplers, etc.).
          int OMPThreadId = omp_get_thread_num();
		  GPMThreadData* threadData = m_threadData[OMPThreadId];

		  // Image pointers
		  Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
		  std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[blockIdx];
          
          // Calculate flux arriving at the gather points 
		  for (size_t i = 0; i < gatherPoints.size(); ++i) {
			  GatherPoint &gp = gatherPoints[i];

              // In case we want to render direct rendering
              // we need to handle in a special case if the current gather point
              // hit the light source.
              if (m_config.directTracing) {
                  if (!gp.currEmission.isZero()) {
                      GradientSamplingRecord gRec(scene, &gp, m_config, *threadData);
                      gRec.estimateGradEmitter(); // directly add the radiance inside the gather point
                  }
              }

              // If we have generated correctly the gather point (on a valid "diffuse" surface)
			  if (gp.depth != -1) {
                  // Query the photon map and estimate the gradient.
                  // For more information, look at gpm_photonmap.{h,cpp} files.
                  GradientSamplingRecord gRec(scene, &gp, m_config, *threadData);
				  Float M = (Float) photonMap->estimateRadianceGrad(gRec, m_config.minDepth, m_config.maxDepth);
			  
                  // Update statistic of the gather point
                  Float N = gp.N;
                  if (N + M == 0)
                      continue;
                  Float ratio = (N + m_config.alpha * M) / (N + M);
				  gp.scale = gp.scale * std::sqrt(ratio);

				  // Update flux of the base gather point (the created one)
                  gp.baseFlux += gRec.baseFlux;
                  gp.baseFlux /= gp.radius * gp.radius * M_PI; // Normalize it

				  // Update flux on the shift paths
                  for (int k = 0; k < 4; ++k) {
                      gp.shiftedFlux[k] += gRec.shiftedFlux[k];
                      gp.shiftedFlux[k] /= gp.radius * gp.radius * M_PI;

                      gp.weightedBaseFlux[k] += gRec.weightedBaseFlux[k];
                      gp.weightedBaseFlux[k] /= gp.radius * gp.radius * M_PI;
                  }
                  gp.N = N + m_config.alpha * M;
			  }

              // Fill the image
			  target[gp.pos.y * m_bitmap->getWidth() + gp.pos.x] = gp.baseFlux / ((Float)m_totalEmitted) + (gp.emission / it);
		  }
	  }
      Log(EInfo, "Gathering done.");

      /* Display gradient image combined (for debug) */
	  if(m_config.showAbsGrad) {
	      ref<Bitmap> gXBitmap = m_bitmap->clone();
	      ref<Bitmap> gYBitmap = m_bitmap->clone();
	      computeGradient(it, cropSize, gXBitmap.get(), gYBitmap.get(), false);

		  ref<Bitmap> gAbsCombined = m_bitmap->clone();
		  Spectrum *targetGX = (Spectrum *) gXBitmap->getUInt8Data();
		  Spectrum *targetGY = (Spectrum *) gYBitmap->getUInt8Data();
		  Spectrum *targetGComb = (Spectrum *) gAbsCombined->getUInt8Data();
		  for(int y = 0; y < cropSize.y; ++y) {
			  for (int x = 0; x < cropSize.x; ++x) {
                  int index = y * gXBitmap->getWidth() + x;
				  targetGComb[index] = (targetGX[index].abs() + targetGY[index].abs()) * 0.5f;
              }
		  }
		  film->setBitmap(gAbsCombined);
          queue->signalRefresh(job);
	  } 

	  // Generate step hdr files
      if (m_config.dumpIteration > 0 && it % m_config.dumpIteration == 0) {
          /* Create bitmap to display gradient images */
          SLog(EInfo, "Saving images...");
          ref<Bitmap> sppmBitmap = m_bitmap->clone();
          addDirectImage(it, cropSize, sppmBitmap);

          ref<Bitmap> gXBitmap = m_bitmap->clone();
          ref<Bitmap> gYBitmap = m_bitmap->clone();
          computeGradient(it, cropSize, gXBitmap.get(), gYBitmap.get(), false);

          if (m_config.pretendSPPM) {
              // A special mode to only output throughput images
              develop(scene, film, sppmBitmap, it);     

              film->setBitmap(m_bitmap);
              queue->signalRefresh(job);
          }
          else {

              // Develop all films
              ref<Bitmap> gXBitmapAbs = gXBitmap->clone();
              ref<Bitmap> gYBitmapAbs = gYBitmap->clone();
              computeAbs(cropSize, gXBitmapAbs);
              computeAbs(cropSize, gYBitmapAbs);
              develop(scene, film, gXBitmapAbs, it, "_dxAbs_");
              develop(scene, film, gYBitmapAbs, it, "_dyAbs_");
              develop(scene, film, sppmBitmap, it);     // Save throughput + direct image

              // Make a reconstruction by solving the Poisson equation
              ref<Bitmap> reconsBitmap = m_bitmapRecons;
              int w = reconsBitmap->getSize().x;
              int h = reconsBitmap->getSize().y;

              /* Transform the data for the solver. */
              size_t subPixelCount = 3 * film->getCropSize().x * film->getCropSize().y;
              std::vector<float> throughputVector(subPixelCount);
              std::vector<float> dxVector(subPixelCount);
              std::vector<float> dyVector(subPixelCount);
              std::vector<float> directVector(subPixelCount);
              std::vector<float> reconstructionVector(subPixelCount);

              std::transform(m_bitmap->getFloatData(), m_bitmap->getFloatData() + subPixelCount, throughputVector.begin(), [](Float x) { return (float)x; });
              std::transform(gXBitmap->getFloatData(), gXBitmap->getFloatData() + subPixelCount, dxVector.begin(), [](Float x) { return (float)x; });
              std::transform(gYBitmap->getFloatData(), gYBitmap->getFloatData() + subPixelCount, dyVector.begin(), [](Float x) { return (float)x; });

              /* NaN check */
              if (m_config.nanCheck) {
                  for (int i = 0; i < subPixelCount; ++i) {
                      if (!std::isfinite(dxVector[i])) {
                          int y = (i / 3) / film->getCropSize().x;
                          int x = (i / 3) % film->getCropSize().x;
                          int c = i % 3;
                          SLog(EWarn, "Gradient x at pixel (%d, %d, %d) is NaN", x, y, c);
                          dxVector[i] = 0.f;
                      }

                      if (!std::isfinite(dyVector[i])) {
                          int y = (i / 3) / film->getCropSize().x;
                          int x = (i / 3) % film->getCropSize().x;
                          int c = i % 3;
                          SLog(EWarn, "Gradient y at pixel (%d, %d, %d) is NaN", x, y, c);
                          dyVector[i] = 0.f;
                      }

                      if (!std::isfinite(throughputVector[i])) {
                          int y = (i / 3) / film->getCropSize().x;
                          int x = (i / 3) % film->getCropSize().x;
                          int c = i % 3;
                          SLog(EWarn, "Throughput at pixel (%d, %d, %d) is NaN", x, y, c);
                          throughputVector[i] = 0.f;
                      }
                  }
              }

              /* Reconstruct. */
              poisson::Solver::Params paramsL1;
              paramsL1.setConfigPreset("L1D");
              poisson::Solver::Params paramsL2;
              paramsL2.setConfigPreset("L2D"); // Use "L2Q" for better reconstruction

              paramsL1.alpha = (float)m_config.reconstructAlpha;
              paramsL1.setLogFunction(poisson::Solver::Params::LogFunction([](const std::string& message) { SLog(EInfo, "%s", message.c_str()); }));
              paramsL2.alpha = (float)m_config.reconstructAlpha;
              paramsL2.setLogFunction(poisson::Solver::Params::LogFunction([](const std::string& message) { SLog(EInfo, "%s", message.c_str()); }));

              /////////////// L2 reconstruction
              if (m_config.reconstructL2) {
                  SLog(EInfo, "Doing L2 reconstruction");
                  poisson::Solver solverL2(paramsL2);
                  solverL2.importImagesMTS(dxVector.data(), dyVector.data(), throughputVector.data(), directVector.data(), film->getCropSize().x, film->getCropSize().y);
                  solverL2.setupBackend();
                  solverL2.solveIndirect();
                  solverL2.exportImagesMTS(reconstructionVector.data());

                  for (int y = 0, p = 0; y < h; ++y) {
                      for (int x = 0; x < w; ++x, p += 3) {
                          Float color[3] = { (Float)reconstructionVector[p], (Float)reconstructionVector[p + 1], (Float)reconstructionVector[p + 2] };
                          reconsBitmap->setPixel(Point2i(x, y), Spectrum(color));
                      }
                  }

                  // Beautify the result
                  clampReconstructionBitmap(it);
                  clampNegative(cropSize, reconsBitmap);
                  addDirectImage(it, cropSize, reconsBitmap);

                  if (m_config.reconstructL1) {
                      develop(scene, film, reconsBitmap, it, "_L2_");
                  }
                  else {
                      develop(scene, film, reconsBitmap, it, "_recons_");
                  }

                  // Show the results if it is needed
                  if (!m_config.showOnlyRecons) {
                      film->setBitmap(m_bitmap);
                      queue->signalRefresh(job);
                  }
                  else {
                      film->setBitmap(m_bitmapRecons);
                      queue->signalRefresh(job);
                  }
              }

              /////////////// L1 reconstruction
              if (m_config.reconstructL1) {
                  SLog(EInfo, "Doing L1 reconstruction");
                  poisson::Solver solverL1(paramsL1);
                  solverL1.importImagesMTS(dxVector.data(), dyVector.data(), throughputVector.data(), directVector.data(), film->getCropSize().x, film->getCropSize().y);
                  solverL1.setupBackend();
                  solverL1.solveIndirect();
                  solverL1.exportImagesMTS(reconstructionVector.data());

                  for (int y = 0, p = 0; y < h; ++y) {
                      for (int x = 0; x < w; ++x, p += 3) {
                          Float color[3] = { (Float)reconstructionVector[p], (Float)reconstructionVector[p + 1], (Float)reconstructionVector[p + 2] };
                          reconsBitmap->setPixel(Point2i(x, y), Spectrum(color));
                      }
                  }

                  // Beautify the result
                  clampReconstructionBitmap(it);
                  clampNegative(cropSize, reconsBitmap);
                  addDirectImage(it, cropSize, reconsBitmap);

                  if (m_config.reconstructL2) {
                      develop(scene, film, reconsBitmap, it, "_L1_");
                  }
                  else {
                      develop(scene, film, reconsBitmap, it, "_recons_");
                  }

                  // Show the results if it is needed
                  if (!m_config.showOnlyRecons) {
                      film->setBitmap(m_bitmap);
                      queue->signalRefresh(job);
                  }
                  else {
                      film->setBitmap(m_bitmapRecons);
                      queue->signalRefresh(job);
                  }
              }
          
          } // End of pretendSPPM

      }
	  // Clean memory
	  SLog(EInfo, "Free memory allocated for light paths...");
	  photonMap->clean(memoryPools);
	  for(MemoryPool& pool: memoryPools) {
		  if(!pool.unused()) {
			  SLog(EError, pool.toString().c_str());
			  SLog(EError, "Memory leak detected");
		  }
	  }

  }

  void computeGradient(int N, const Vector2i& cropSize, Bitmap* gXBitmap, Bitmap* gYBitmap, bool useAbs = true) {
	  Spectrum *targetGX = (Spectrum *) gXBitmap->getUInt8Data();
	  Spectrum *targetGY = (Spectrum *) gYBitmap->getUInt8Data();
	  for(int y = 0; y < cropSize.y; ++y) {
		  for (int x = 0; x < cropSize.x; ++x) {
			  GatherPoint* curr = m_imgGP[x][y];
			  
              Spectrum gX;
              if (x == cropSize.x - 1) {
                  gX = (curr->shiftedFlux[ERight] - curr->weightedBaseFlux[ERight]);
              }
              else {
                  GatherPoint* right = m_imgGP[x + 1][y];
                  gX = (curr->shiftedFlux[ERight] - curr->weightedBaseFlux[ERight]) +
                       (right->weightedBaseFlux[ELeft] - right->shiftedFlux[ELeft]);
              }
              gX /= ((Float)m_totalEmitted);

              if (m_config.directTracing) {
                  // Gradient caused by non-density estimation path when gather point hits the light
                  if (x == cropSize.x - 1) {
                      gX += (curr->shiftedEmitterFlux[ERight] - curr->weightedEmitterFlux[ERight]) / N;
                  }
                  else {
                      GatherPoint* right = m_imgGP[x + 1][y];
                      gX += ((curr->shiftedEmitterFlux[ERight] - curr->weightedEmitterFlux[ERight]) +
                          (right->weightedEmitterFlux[ELeft] - right->shiftedEmitterFlux[ELeft])) / N;
                  }
              }

              Spectrum gY;
              if (y == cropSize.y - 1) {
                  gY = (curr->shiftedFlux[ETop] - curr->weightedBaseFlux[ETop]);
              }
              else {
                  GatherPoint* top = m_imgGP[x][y + 1];
                  gY = (curr->shiftedFlux[ETop] - curr->weightedBaseFlux[ETop]) +
                       (top->weightedBaseFlux[EBottom] - top->shiftedFlux[EBottom]);
              }
              gY /= ((Float)m_totalEmitted);

              if (m_config.directTracing) {
                  if (y == cropSize.y - 1) {
                      gY += (curr->shiftedEmitterFlux[ETop] - curr->weightedEmitterFlux[ETop]) / N;
                  }
                  else {
                      GatherPoint* top = m_imgGP[x][y + 1];
                      gY += ((curr->shiftedEmitterFlux[ETop] - curr->weightedEmitterFlux[ETop]) +
                          (top->weightedEmitterFlux[EBottom] - top->shiftedEmitterFlux[EBottom])) / N;
                  }
              }

			  if(useAbs) {
                  targetGX[y * gXBitmap->getWidth() + x] = gX.abs();
                  targetGY[y * gYBitmap->getWidth() + x] = gY.abs();
			  } else {
                  targetGX[y * gXBitmap->getWidth() + x] = gX;
                  targetGY[y * gYBitmap->getWidth() + x] = gY;
			  }
		  }
	  }
  }

  void computeGradient(const Vector2i& cropSize, Bitmap* gXPositive, Bitmap* gXNegative, Bitmap* gYPositive, Bitmap* gYNegative, bool useAbs = true) {
      Spectrum *gradXPositive = (Spectrum *)gXPositive->getUInt8Data();
      Spectrum *gradXNegative = (Spectrum *)gXNegative->getUInt8Data();
      Spectrum *gradYPositive = (Spectrum *)gYPositive->getUInt8Data();
      Spectrum *gradYNegative = (Spectrum *)gYNegative->getUInt8Data();
      for (int y = 0; y < cropSize.y; ++y) {
          for (int x = 0; x < cropSize.x; ++x) {
              GatherPoint* curr = m_imgGP[x][y];

              Spectrum gX0(0.0f), gX1(0.0f), gY0(0.0f), gY1(0.0f);

              gX0 = (curr->shiftedFlux[ERight] - curr->weightedBaseFlux[ERight]);
              gX0 /= ((Float)m_totalEmitted);

              if (x != cropSize.x - 1) {
                  GatherPoint* right = m_imgGP[x + 1][y];
                  gX1 = (right->weightedBaseFlux[ELeft] - right->shiftedFlux[ELeft]);
                  gX1 /= ((Float)m_totalEmitted);
              }

              gY0 = (curr->shiftedFlux[ETop] - curr->weightedBaseFlux[ETop]);
              gY0 /= ((Float)m_totalEmitted);

              if (y != cropSize.y - 1) {
                  GatherPoint* top = m_imgGP[x][y + 1];
                  gY1 = (top->weightedBaseFlux[EBottom] - top->shiftedFlux[EBottom]);
                  gY1 /= ((Float)m_totalEmitted);
              }

              if (useAbs) {
                  gX0 = gX0.abs();
                  gX1 = gX1.abs();
                  gY0 = gY0.abs();
                  gY1 = gY1.abs();
              }

              gradXPositive[y * gXPositive->getWidth() + x] = gX0;
              gradXNegative[y * gXNegative->getWidth() + x] = gX1;

              gradYPositive[y * gYPositive->getWidth() + x] = gY0;
              gradYNegative[y * gYNegative->getWidth() + x] = gY1;
          }
      }
  }

  void computeAbs(const Vector2i& cropSize, Bitmap* bitmap) {
	  Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
	  for(int y = 0; y < cropSize.y; ++y) {
		  for (int x = 0; x < cropSize.x; ++x) {
			  target[y * bitmap->getWidth() + x] = target[y * bitmap->getWidth() + x].abs();
		  }
	  }
  }

  void clampNegative(const Vector2i& cropSize, Bitmap *bitmap) {
    Spectrum *target = (Spectrum *) bitmap->getUInt8Data();
    for(int y = 0; y < cropSize.y; ++y) {
 	    for (int x = 0; x < cropSize.x; ++x) {
		    target[y * bitmap->getWidth() + x].clampNegative();
	    }
    }
  }

  /**
   * Avoid energy diffusion into empty region that has no valid gather points, 
   * which can cause issues in relative error calculation.
   */
  void clampReconstructionBitmap(int N) {
    ref<Bitmap> throughput = m_bitmap;
    for (int blockIdx = 0; blockIdx < (int)m_gatherBlocks.size(); ++blockIdx) {
        std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[blockIdx];

        for (size_t i = 0; i < gatherPoints.size(); ++i) {
            const GatherPoint &gp = gatherPoints[i];
            if (m_config.forceBlackPixels && gp.depth == -1 &&
                throughput->getPixel(gp.pos).isZero()) 
            {
				m_bitmapRecons->setPixel(gp.pos, gp.emission / N);
            }
        }
    }
  }

  void addDirectImage(int N, const Vector2i& cropSize, Bitmap *bitmap) {
    for (int blockIdx = 0; blockIdx < (int)m_gatherBlocks.size(); ++blockIdx) {
        std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[blockIdx];

        for (size_t i = 0; i < gatherPoints.size(); ++i) {
            const GatherPoint &gp = gatherPoints[i];
            
            if (!gp.directEmission.isZero()) {
                Spectrum pixel = bitmap->getPixel(gp.pos);
                bitmap->setPixel(gp.pos, pixel + gp.directEmission / N);
            }
        }
    }
  }

  void getDirectImage(int N, const Vector2i& cropSize, Bitmap *bitmap) {
    for (int blockIdx = 0; blockIdx < (int)m_gatherBlocks.size(); ++blockIdx) {
        std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[blockIdx];

        for (size_t i = 0; i < gatherPoints.size(); ++i) {
            const GatherPoint &gp = gatherPoints[i];
            bitmap->setPixel(gp.pos, gp.directEmission / N);
        }
    }
  }
  

  void develop(Scene* scene, Film* film, const Bitmap* bitmap,
               int currentIteration, const std::string& suffixName = "_") {
	std::stringstream ss;
	ss << scene->getDestinationFile().string() << suffixName
	<< currentIteration;
	std::string path = ss.str();
	film->setBitmap(bitmap);
	film->setDestinationFile(path, 0);
	film->develop(scene, 0.f);
  }

	std::string toString() const {
		std::ostringstream oss;
		oss << "SPPMIntegrator[" << endl
            << "  maxDepth = " << m_config.maxDepth << "," << endl
            << "  rrDepth = " << m_config.rrDepth << "," << endl
            << "  initialRadius = " << m_config.initialRadius << "," << endl
            << "  alpha = " << m_config.alpha << "," << endl
            << "  photonCount = " << m_config.photonCount << "," << endl
            << "  granularity = " << m_config.granularity << "," << endl
            << "  maxPasses = " << m_config.maxPasses << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
  	std::vector<GPMThreadData*> m_threadData;
	std::vector<std::vector<GatherPoint> > m_gatherBlocks;
  	std::vector<std::vector<GatherPoint*>> m_imgGP;
  	RadiusInitializer* m_gpManager;

	std::vector<Point2i> m_offset;
	ref<Mutex> m_mutex;
	ref<Bitmap> m_bitmap;
    ref<Bitmap> m_bitmapGx, m_bitmapGy;
    ref<Bitmap> m_bitmapRecons;

	size_t m_totalEmitted, m_totalPhotons;
	bool m_running;

    GPMConfig m_config;
};

MTS_IMPLEMENT_CLASS_S(GPMIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(GPMIntegrator, "Gradient domain PM");
MTS_NAMESPACE_END
