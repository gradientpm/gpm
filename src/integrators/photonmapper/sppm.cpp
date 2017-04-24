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
#include <mitsuba/render/gatherproc.h>
#include <mitsuba/render/renderqueue.h>

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

/// Represents one individual PPM gather point including relevant statistics
struct GatherPoint {
  Intersection its;
  Float radius;
  Spectrum weight;
  Spectrum flux;
  Spectrum emission;
  Float N;
  int depth;
  Point2i pos;

  // BSDF informations
  int sampledComponent;
  float pdfComponent;
  Float scale;

  inline GatherPoint() : weight(0.0f), flux(0.0f), emission(0.0f), N(0.0f),
  sampledComponent(-1), pdfComponent(1.0f), scale(1.0f), depth(-1) { }

  void resetTemp() {
	  // Nothing to do?
  }

  inline void rescaleRadii(Float v) {
	  flux *= v;
  }

};

MTS_NAMESPACE_END

#include "utilities/initializeRadius.h"

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
class SPPMIntegrator : public Integrator {
public:
	SPPMIntegrator(const Properties &props) : Integrator(props) {
        m_initialScale = props.getFloat("initialScale", 1.f);
		/* Alpha parameter from the paper (influences the speed, at which the photon radius is reduced) */
		m_alpha = props.getFloat("alpha", .7);
		/* Number of photons to shoot in each iteration */
		m_photonCount = props.getInteger("photonCount", 250000);
		/* Granularity of the work units used in parallelizing the
		   particle tracing task (default: choose automatically). */
		m_granularity = props.getInteger("granularity", 0);
		/* Longest visualized path length (<tt>-1</tt>=infinite). When a positive value is
		   specified, it must be greater or equal to <tt>2</tt>, which corresponds to single-bounce
		   (direct-only) illumination */
		m_maxDepth = props.getInteger("maxDepth", -1);
        m_minDepth = props.getInteger("minDepth", 0);
		/* Depth to start using russian roulette */
		m_rrDepth = props.getInteger("rrDepth", 3);
		/* Indicates if the gathering steps should be canceled if not enough photons are generated. */
		m_autoCancelGathering = props.getBoolean("autoCancelGathering", true);
		/* Maximum number of passes to render. -1 renders until the process is stopped. */
		m_maxPasses = props.getInteger("maxPasses", -1);
		m_mutex = new Mutex();
		if (m_maxDepth <= 1 && m_maxDepth != -1)
			Log(EError, "Maximum depth must be set to \"2\" or higher!");
		if (m_maxPasses <= 0 && m_maxPasses != -1)
			Log(EError, "Maximum number of Passes must either be set to \"-1\" or \"1\" or higher!");

		// Create the instance to generate the gather point in the similar way for all
		// PPM/SPPM codes
		m_gpManager = new RadiusInitializer(props);

		m_maxRenderingTime = props.getInteger("maxRenderingTime", INT_MAX);
		m_dumpIteration = props.getInteger("dumpIteration", 5);
		if (m_maxRenderingTime != INT_MAX && m_maxPasses != INT_MAX) {
			Log(EError, "Max pass and time is incompatible!");
		}

		// Debug options / Developement proposes
		m_computeGrad = props.getBoolean("computeGrad", true);
		m_directTracing = props.getBoolean("directTracing", false);
	}

	SPPMIntegrator(Stream *stream, InstanceManager *manager)
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
		const_cast<Scene*>(scene)->weightEmitterFlux();
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

		/* Allocate memory */
		m_bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
		m_bitmap->clear();
		for (int yofs=0; yofs<cropSize.y; yofs += blockSize) {
			for (int xofs=0; xofs<cropSize.x; xofs += blockSize) {
				m_gatherBlocks.push_back(std::vector<GatherPoint>());
				m_offset.push_back(Point2i(cropOffset.x + xofs, cropOffset.y + yofs));
				std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[m_gatherBlocks.size()-1];
				int nPixels = std::min(blockSize, cropSize.y-yofs)
							* std::min(blockSize, cropSize.x-xofs);
				gatherPoints.resize(nPixels);
				for (int i=0; i<nPixels; ++i) {
					gatherPoints[i].radius = 0;
                    gatherPoints[i].scale = m_initialScale;
				}
			}
		}

		// Initialize the class responsible to the GP generation
		// and the radii initialization
		m_gpManager->init(scene, m_maxDepth, m_gatherBlocks, m_offset);

		/* Create a sampler instance for every core */
		std::vector<SerializableObject *> samplers(sched->getCoreCount());
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

		// Create the files to dump information about the rendering
		// Also create timer to track algorithm performance
		std::string timeFilename = scene->getDestinationFile().string()
			+ "_time.csv";
		std::ofstream timeFile(timeFilename.c_str());
		ref<Timer> renderingTimer = new Timer;

		// Rendering loop
		int it = 1;
		while (m_running && (m_maxPasses == -1 || it < m_maxPasses)) {
			Log(EInfo, "Regenerating gather points positions and radius!");
			m_gpManager->regeneratePositionAndRadius();
			Log(EInfo, "Done regenerating!");
			m_gpManager->rescaleFlux();

			if (it == 1) {

				//FIXME: Change this weird behavior. Indeed gp.pos is set during the gp tracing...
				/* Create a copy of the gather point using image space,
                 * to be able to retrive them easily
                 */
				m_imgGP.resize(cropSize.x);
				for (int x=0; x<cropSize.x; x += 1) {
					m_imgGP[x].resize(cropSize.y);
				}
				for(int idBlock = 0; idBlock < m_gatherBlocks.size(); idBlock++) {
					std::vector<GatherPoint>& currBlock = m_gatherBlocks[idBlock];
					for(GatherPoint& gp: currBlock) {
						m_imgGP[gp.pos.x][gp.pos.y] = &gp;
					}
				}
			}

			photonMapPass(it, queue, job, film, sceneResID,
					sensorResID, samplerResID, scene);

			// Write down some
			if((it % m_dumpIteration) == 0) {
				develop(scene, film, m_bitmap, it, "_");
			}

			// === Update the log time
			unsigned int milliseconds = renderingTimer->getMilliseconds();
			timeFile << (milliseconds / 1000.f) << ",\n";
			timeFile.flush();
			Log(EInfo, "Rendering time: %i, %i", milliseconds / 1000,
				milliseconds % 1000);
			m_maxRenderingTime -= (milliseconds / 1000);
			if (m_maxRenderingTime < 0) {
				m_running = false;
				Log(EInfo, "Max time reaching !");
			}

			// === update variables && free memory
			renderingTimer->reset();


			++it;
		}

#ifdef MTS_DEBUG_FP
		disableFPExceptions();
#endif

		for (size_t i=0; i<samplers.size(); ++i)
			samplers[i]->decRef();

		sched->unregisterResource(samplerResID);
		return true;
	}

	void photonMapPass(int it, RenderQueue *queue, const RenderJob *job,
			Film *film, int sceneResID, int sensorResID, int samplerResID, Scene* scene) {
		Log(EInfo, "Performing a photon mapping pass %i (" SIZE_T_FMT " photons so far)",
				it, m_totalPhotons);
		ref<Scheduler> sched = Scheduler::getInstance();
		Vector2i cropSize = film->getCropSize();

		/* Generate the global photon map */
		ref<GatherPhotonProcess> proc = new GatherPhotonProcess(
			GatherPhotonProcess::EAllSurfacePhotons, m_photonCount,
			m_granularity, m_maxDepth == -1 ? -1 : m_maxDepth-1, m_rrDepth, true,
			m_autoCancelGathering, job);

		proc->bindResource("scene", sceneResID);
		proc->bindResource("sensor", sensorResID);
		proc->bindResource("sampler", samplerResID);

		sched->schedule(proc);
		sched->wait(proc);

		ref<PhotonMap> photonMap = proc->getPhotonMap();
		photonMap->build();
		Log(EDebug, "Photon map full. Shot " SIZE_T_FMT " particles, excess photons due to parallelism: "
			SIZE_T_FMT, proc->getShotParticles(), proc->getExcessPhotons());

		Log(EInfo, "Gathering ..");
		m_totalEmitted += proc->getShotParticles();
		m_totalPhotons += photonMap->size();
		film->clear();
		#if defined(MTS_OPENMP)
			#pragma omp parallel for schedule(dynamic)
		#endif
		for (int blockIdx = 0; blockIdx<(int) m_gatherBlocks.size(); ++blockIdx) {
			std::vector<GatherPoint> &gatherPoints = m_gatherBlocks[blockIdx];

			Spectrum *target = (Spectrum *) m_bitmap->getUInt8Data();
			for (size_t i=0; i<gatherPoints.size(); ++i) {
				GatherPoint &gp = gatherPoints[i];
				Spectrum contrib(0.0f);

				if (gp.depth != -1) {
                    
                    Spectrum flux(0.0f);

                    Float M = (Float)photonMap->estimateRadianceGP(
                                        gp.its, gp.radius, flux, 
                                        gp.depth, 
                                        m_minDepth,
                                        m_maxDepth == -1 ? INT_MAX : m_maxDepth,
                                        gp.sampledComponent);

                    // Update GP
                    Float N = gp.N;
                    if (N + M != 0) {
                      Float ratio = (N + m_alpha * M) / (N + M);
                      gp.scale = gp.scale * std::sqrt(ratio);
                      gp.flux = (gp.flux + gp.weight * flux);
                      gp.flux /= gp.radius*gp.radius * M_PI; // become radiance
                      gp.N = N + m_alpha * M;
                    }
                }

                // When depth == -1, gp.flux is not rescaled and is still radiance value. No need to divide radius.
                contrib = gp.flux / ((Float)m_totalEmitted);
                if(m_directTracing) {
                  contrib += gp.emission / it; // Monte carlo estimator
                }

                target[gp.pos.y * m_bitmap->getWidth() + gp.pos.x] = contrib;
			}
		}

		if(m_computeGrad && (it % m_dumpIteration) == 0) {
			// Compute gradient using finite differences
			ref<Bitmap> gXBitmap = m_bitmap->clone();
			ref<Bitmap> gYBitmap = m_bitmap->clone();

			computeGradientFinite(cropSize, gXBitmap.get(), gYBitmap.get());

			develop(scene, film, gXBitmap, it, "_dxAbs_");
			develop(scene, film, gYBitmap, it, "_dyAbs_");
		}


		film->setBitmap(m_bitmap);
		queue->signalRefresh(job);
	}

  void develop(Scene* scene, Film* film, Bitmap* bitmap,
			   int currentIteration, const std::string& suffixName = "_") {
	  std::stringstream ss;
	  ss << scene->getDestinationFile().string() << suffixName
		 << currentIteration;
	  std::string path = ss.str();

	  film->setBitmap(bitmap);
	  film->setDestinationFile(path, 0);
	  film->develop(scene, 0.f);

  }

  void computeGradientFinite(const Vector2i& cropSize, Bitmap* gXBitmap, Bitmap* gYBitmap, bool useAbs = true) {
	  Spectrum *targetGX = (Spectrum *) gXBitmap->getUInt8Data();
	  Spectrum *targetGY = (Spectrum *) gYBitmap->getUInt8Data();
	  for(int y = 1; y < cropSize.y-1; ++y) { 
		  for (int x = 1; x < cropSize.x-1; ++x) {
			  GatherPoint* curr = m_imgGP[x][y];
			  GatherPoint* right = m_imgGP[x+1][y];
			  GatherPoint* top = m_imgGP[x][y+1];

			  Spectrum gX = (right->flux - curr->flux)  / ((Float) m_totalEmitted);
			  Spectrum gY = (top->flux - curr->flux)  / ((Float) m_totalEmitted);

			  if(useAbs) {
				  targetGX[y * m_bitmap->getWidth() + x] = gX.abs();
				  targetGY[y * m_bitmap->getWidth() + x] = gY.abs();
			  } else {
				  targetGX[y * m_bitmap->getWidth() + x] = gX;
				  targetGY[y * m_bitmap->getWidth() + x] = gY;
			  }
		  }
	  }
  }

	std::string toString() const {
		std::ostringstream oss;
		oss << "SPPMIntegrator[" << endl
			<< "  maxDepth = " << m_maxDepth << "," << endl
			<< "  rrDepth = " << m_rrDepth << "," << endl
			<< "  alpha = " << m_alpha << "," << endl
			<< "  photonCount = " << m_photonCount << "," << endl
			<< "  granularity = " << m_granularity << "," << endl
			<< "  maxPasses = " << m_maxPasses << endl
			<< "]";
		return oss.str();
	}

	MTS_DECLARE_CLASS()
private:
	std::vector<std::vector<GatherPoint> > m_gatherBlocks;
  	RadiusInitializer* m_gpManager;

	std::vector<Point2i> m_offset;
	ref<Mutex> m_mutex;
	ref<Bitmap> m_bitmap;
    Float m_initialScale;
    Float m_alpha;
	int m_photonCount, m_granularity;
	int m_maxDepth, m_minDepth, m_rrDepth;
	size_t m_totalEmitted, m_totalPhotons;
	bool m_running;
	bool m_autoCancelGathering;
	int m_maxPasses;

  	// A structure to easy gather neighbors pixels
  	std::vector<std::vector<GatherPoint*>> m_imgGP;
  	int m_dumpIteration;
  	bool m_computeGrad;
  	int m_maxRenderingTime;
  	bool m_directTracing;
};

MTS_IMPLEMENT_CLASS_S(SPPMIntegrator, false, Integrator)
MTS_EXPORT_PLUGIN(SPPMIntegrator, "Stochastic progressive photon mapper");
MTS_NAMESPACE_END
