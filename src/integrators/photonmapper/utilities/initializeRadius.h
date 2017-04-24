#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

#define USE_BBOX_INITIALRADIUS 1

class RadiusInitializer {
 public:
  RadiusInitializer(const Properties &props)
      : scene(0),
        maxDepth(-1) {

    referenceMod = props.getBoolean("referenceMod", false);
    m_bounceRoughness = props.getFloat("bounceRoughness");
    m_directTracing = props.getBoolean("directTracing", false);
    if (m_bounceRoughness <= 0.0 || m_bounceRoughness > 1.0) {
      SLog(EError, "Bad roughtness constant: %f", m_bounceRoughness);
    }
  }

  virtual ~RadiusInitializer() {
  }

  virtual void init(Scene *scene, int maxDepth,
                    std::vector<std::vector<GatherPoint>> &gatherPoints,
                    const std::vector<Point2i> &offset) {
    SLog(EInfo, "Initialize GP generation object");

    // Copy the object
    // These will be used to generate
    // the gather points through the scene
    this->scene = scene;
    this->maxDepth = maxDepth;
    this->gatherBlocks = &gatherPoints;
    this->offset = &offset;
    this->techniques = techniques;

    // And then initialise the samplers
    // without shift in the random
    // sequence
    generateSamplers(0);
  }

  void generateSamplers(int shiftRandomNumber) {
    // Initialize samplers for generate GP
    // Each pixel blocks will have is own
    // sampler to generate the same set
    // of gather points

    Properties props("independent");
    if (referenceMod) {
      props.setBoolean("randInit", true);
    }

    // --- Sampler to generate seed of other
    ref<Sampler> samplerIndependent =
        static_cast<Sampler *>(PluginManager::getInstance()->createObject(
            MTS_CLASS(Sampler), props));

    if (shiftRandomNumber != 0) {
      SLog(EInfo, "Make an shift of random number generator: %i (%i)",
           shiftRandomNumber, offset->size());
      // Make the shift by advancing in the
      // sequence by calling the
      for (size_t i = 0; i < (offset->size() * shiftRandomNumber); ++i) {
        samplerIndependent->next2D();
      }
    }

    // --- Create all samplers
    samplers.resize(offset->size());  //< Create sampler
    for (size_t i = 0; i < offset->size(); ++i) {
      ref<Sampler> clonedIndepSampler = samplerIndependent->clone();
      samplers[i] = clonedIndepSampler;
    }
  }

  virtual Float getRadiusRayDifferential(RayDifferential &ray,
                                         Float totalDist) {

    if (ray.hasDifferentials) {  // nbComponentExtra == 0 &&
      ray.scaleDifferential(1.f);  
      Point posProj = ray.o + ray.d * totalDist;
      Point rX = ray.rxOrigin + ray.rxDirection * totalDist;
      Point rY = ray.ryOrigin + ray.ryDirection * totalDist;
      Float dX = (rX - posProj).length();
      Float dY = (rY - posProj).length();

      Float r = std::max(dX, dY);
      if (r > 100) {
        SLog(EError, "Infinite radius %f", r);
      }
      return r;
    } else {
      SLog(EError, "No ray differential");
      return 0.f;
    }
  }

  void regeneratePositionAndRadius() {
    // Get some data ...
    ref<Sensor> sensor = scene->getSensor();
    bool needsApertureSample = sensor->needsApertureSample();
    bool needsTimeSample = sensor->needsTimeSample();
    ref<Film> film = sensor->getFilm();
    Vector2i cropSize = film->getCropSize();
    Point2i cropOffset = film->getCropOffset();
    int blockSize = scene->getBlockSize();

#if defined(MTS_OPENMP)
    ref<Scheduler> sched = Scheduler::getInstance();
    size_t nCores = sched->getCoreCount();
    Thread::initializeOpenMP(nCores);
#endif

#if defined(MTS_OPENMP)
    //  schedule(dynamic) removed
#pragma omp parallel for
#endif
    for (int i = 0; i < (int) gatherBlocks->size(); ++i) {  // For all gather points
      // Get the sampler associated
      // to the block
      ref<Sampler> sampler = samplers[i];

      // For all the gather points int the block
      // be carefull of the offset of the image plane
      std::vector<GatherPoint> &gb = (*gatherBlocks)[i];
      int xofs = (*offset)[i].x, yofs = (*offset)[i].y;
      int index = 0;
      for (int yofsInt = 0; yofsInt < blockSize; ++yofsInt) {
        if (yofsInt + yofs - cropOffset.y >= cropSize.y)
          continue;
        for (int xofsInt = 0; xofsInt < blockSize; ++xofsInt) {
          if (xofsInt + xofs - cropOffset.x >= cropSize.x)
            continue;

          // Get the gather point and clear it
          // (temp data erased) + prepare data
          // to sample the gp position
          Point2 apertureSample, sample;
          Float timeSample = 0.0f;
          GatherPoint &gatherPoint = (gb[index++]);
          gatherPoint.resetTemp(); // Reset M and collected flux associated to GP

          // Initialize the GP plane position
          gatherPoint.pos = Point2i(xofs + xofsInt, yofs + yofsInt);

          // (sample): Randomly select the pixel position
          // Special case to the first pass where the gp
          // are sent in the middle of the pixel
          // to compute the associated radii
          if (needsApertureSample)
            apertureSample = Point2(0.5f);
          if (needsTimeSample)
            timeSample = 0.5f;

          sample = sampler->next2D();
          sample += Vector2((Float) gatherPoint.pos.x,
                            (Float) gatherPoint.pos.y);

          // Sample the primary ray from the camera
          RayDifferential rayCamera;
          sensor->sampleRayDifferential(rayCamera, sample, apertureSample,
                                        timeSample);
          RayDifferential ray = rayCamera;

          // Initialize data to bounce
          // through the scene
          Spectrum weight(1.0f);
          int depth = 1;
          gatherPoint.radius = 1;
          //gatherPoint.emission = Spectrum(0.f);
          Float traveledDistance = 0.f;  //< The total distance traveled

          // Invalid - for now
          gatherPoint.depth = -1;

          // Bounce GP in the scene
          while ((depth < maxDepth || maxDepth == -1) && (!weight.isZero())) {

            // Bounce or send primary ray
            scene->rayIntersect(ray, gatherPoint.its);
            if (gatherPoint.its.isValid()) {
              if(m_directTracing && gatherPoint.its.isEmitter()) {
                gatherPoint.emission += weight * gatherPoint.its.Le(-ray.d);
              }
            } else {
              if(m_directTracing) {
                // If no intersection, check the envmap
                gatherPoint.emission += weight * scene->evalEnvironment(ray);
              }
              break; // No gather point here
            }

            // Compute radius using the total distance.
            traveledDistance += gatherPoint.its.t;

            // Get bsdf at intersection
            // Sample randomly one direction
            const BSDF *bsdf = gatherPoint.its.getBSDF();
            BSDFSamplingRecord bRec(gatherPoint.its, sampler);


            // Make the bounce decision
            // This constant will be used for:
            //  - component selection
            //  - bounce decision

            // Select one component
            Float pdfComponent = 1.f;
            Point2 randomBSDFSample = sampler->next2D();
            int componentSelected = bsdf->sampleComponent(bRec, pdfComponent,
                                                      randomBSDFSample,
                                                      m_bounceRoughness);


            // Check the PDF for sample an component
            // There is a problem when it's equal to 0
            // Arrives when ray leaking in RoughtPlastic
            if (pdfComponent == 0) {
              //SLog(EWarn, "Component selection failed");
              // Debug
              if (componentSelected != -1) {
                SLog(
                    EError,
                    "Not the good component is returned for component selection");
              }
              break;
            }

            // Sanity check
            // -1 for component selected = same behavior for all the bsdf component
            // in this case, the probability to pick all need to be 1
            if (componentSelected == -1 && pdfComponent != 1.f) {
              SLog(EError, "All component is selected by the pdf is not 1");
            }

            // Query the roughness to know the bounce decision
            // if the componentSelected is all, the bounce decision
            // will be the same for all component, so take one of them
            // to query the roughness
            int componentQuery = (
                componentSelected == -1 ? 0 : componentSelected);
            bool roughFunction = bsdf->getRoughness(gatherPoint.its,
                                                componentQuery)
                >= m_bounceRoughness;

            // Sample with the selected component only
            bRec.component = componentSelected;

            Spectrum bsdfValue = bsdf->sample(bRec, sampler->next2D());
            roughFunction &= !(bRec.sampledType & BSDF::EDelta);

            // If the BSDF on the gather point is rough enough
            // We create a gather point here
            if(roughFunction) {
              
              // Pre-multiply pdfComponent into weight, since Mitsuba's photon evaluation 
              // does not take in a gather point structure
              weight /= pdfComponent;

              gatherPoint.depth = depth;
              gatherPoint.weight = weight;
              gatherPoint.pdfComponent = pdfComponent;
              gatherPoint.sampledComponent = componentSelected;

              // Compute the radius using the ray differential if this is the first gather point
              // Note that this radius is rescale by the scale factor of the gatherpoint
              if(gatherPoint.scale == 0.f) {
                SLog(EError, "Zero scale on valid gather point");
              }
              float radius = getRadiusRayDifferential(rayCamera, traveledDistance);
              gatherPoint.radius = radius * gatherPoint.scale;

              break; // Finish, we have generated a gather point
            }

            // This are the vectors for bouncing if necessary
            Vector wo = gatherPoint.its.toWorld(bRec.wo);

            // Problem in the BSDF sampling,
            if (bsdfValue.isZero()) {
              // This can happens was we have shading normals.  
              break;
            }

            /* Trace a ray in this direction */
            // Update the bouncing ray
            // and update the depth of the GP
            ray = RayDifferential(gatherPoint.its.p,
                                  wo,
                                  ray.time);

            // Update the weight associated
            // to the GP
            weight *= bsdfValue / pdfComponent;
            ++depth;

            // Russian roulette decision
            if (depth > 10) {
              Float q = std::min(weight.max(), (Float) 0.95f);
              if (sampler->next1D() >= q) {
                // RR decided to stop here
                // make invalid GP
                break;
              }
              // Scale to take into account RR
              weight /= q;
            }

          } // End of while(true)

          sampler->advance();
        }
      }
    }  // End of the sampling

    //////////////////////////////////
    // Display extra informations
    // And in case not initialized GP
    // gets the maximum radii
    //////////////////////////////////
    // === Find max Size
    Float radiusMax = 0;
    Float radiusMin = std::numeric_limits<Float>::max();
    Float radiusAvg = 0;
    int k = 0;
    for (size_t j = 0; j < gatherBlocks->size(); j++) {
      std::vector<GatherPoint> &gb = (*gatherBlocks)[j];
      for (size_t i = 0; i < gb.size(); ++i) {
        GatherPoint &gp = gb[i];
        radiusMax = std::max(radiusMax, gp.radius);
        radiusMin = std::min(radiusMin, gp.radius);
        radiusAvg += gp.radius;
        k++;
      }
    }
    radiusAvg /= k;
    SLog(EInfo, "Max radius: %f", radiusMax);
    SLog(EInfo, "Min radius: %f", radiusMin);
    SLog(EInfo, "Avg radius: %f", radiusAvg);
  }

  void rescaleFlux() {
    for (size_t j = 0; j < gatherBlocks->size(); j++) {
      std::vector<GatherPoint> &gb = (*gatherBlocks)[j];
      for (size_t i = 0; i < gb.size(); ++i) {
        GatherPoint &gp = gb[i];
        if (gp.depth != -1 && gp.its.isValid()) {
          if (gp.radius == 0) {
            // No radius => Error because we will loose the flux
            SLog(EError, "Valid GP with null radius");
          } else {
            gp.rescaleRadii(gp.radius * gp.radius * M_PI);
          }

        } else {

        }
      }
    }
  }

 protected:

  void resetInitialRadius(Float initialRadius) {
    SLog(EInfo, "Reset Initial radius to: %f", initialRadius);
    for (size_t j = 0; j < gatherBlocks->size(); j++) {
      std::vector<GatherPoint> &gb = (*gatherBlocks)[j];

      // === Create & Initialize all gather points in the block
      for (size_t i = 0; i < gb.size(); ++i) {
        GatherPoint &gp = gb[i];
        gp.radius = initialRadius;
      }
    }
  }

 protected:
  // === Config attributs
  Scene *scene;
  int maxDepth;

  // === Gather blocks attributes
  std::vector<std::vector<GatherPoint>> *gatherBlocks;
  const std::vector<Point2i> *offset;

  // === Sampler attributes
  ref_vector <Sampler> samplers;

  // In the reference mode, the sampler are initialized
  // randomly to be sure to not have the same
  // sequence of gather points generated
  bool referenceMod;

  // Used techniques
  int techniques;

  // Bounce constant decision
  Float m_bounceRoughness;
  bool m_directTracing;
};

MTS_NAMESPACE_END
