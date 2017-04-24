#pragma once

#include <mitsuba/mitsuba.h>
#include <mitsuba/render/scene.h>

#if defined(MTS_OPENMP)
# include <omp.h>
#endif

MTS_NAMESPACE_BEGIN

static inline Float geometryOpposingTerm(const Point &p, const Point &q, const Vector &qn) {
    Vector pq = q - p;
    return absDot(qn, normalize(pq)) / pq.lengthSquared();
}

static inline Float geometryOpposingTerm(const Path &path, int b, int c) {
  Point p = path.vertex(b)->getPosition();
  Point q = path.vertex(c)->getPosition();
  Vector qn = path.vertex(c)->getGeometricNormal();
  Vector pq = q - p;
  return absDot(qn, normalize(pq)) / (pq.lengthSquared());
}

static inline Float cosine(const Path &path, int b, int c) {
    // TODO: can use edge->d
    Point p = path.vertex(b)->getPosition();
    Point q = path.vertex(c)->getPosition();
    Vector pn = path.vertex(b)->getGeometricNormal();
    return absDot(pn, normalize(q - p));
}


struct PathEx : public Path {

    /**
     * Attempt to trace a path that ends on the first diffuse vertex. 
     * 
     * When maximum path length is reached, the last vertex might or might not be diffused.
     *
     * \param nSteps    Maximum vertex ID allowed (= expected depth + 1).
     * \return          Vertex ID of the last vertex (= traced depth + 1). 
     */
    int randomWalkFromPixelToFirstDiffuse(const Scene *scene, Sampler *sampler,
        int nSteps, const Point2i &pixelPosition, int rrStart, MemoryPool &pool,
        Float& pdfComponent) {

        PathVertex *v1 = pool.allocVertex(), *v2 = pool.allocVertex();
        PathEdge *e0 = pool.allocEdge(), *e1 = pool.allocEdge();

        /* Use a special sampling routine for the first two sensor vertices so that
        the resulting subpath passes through the specified pixel position */
        int t = vertex(0)->sampleSensor(scene,
            sampler, pixelPosition, e0, v1, e1, v2);

        // t is latest vertex ID

        if (t < 1) {
            pool.release(e0);
            pool.release(v1);
            return 0;
        }

        append(e0, v1);

        if (t < 2) {
            pool.release(e1);
            pool.release(v2);
            return 1;
        }

        append(e1, v2);

        PathVertex *predVertex = v1, *curVertex = v2;
        PathEdge *predEdge = e1;
        Spectrum throughput(1.0f);

        for (; t<nSteps || nSteps == -1; ++t) {
            const BSDF* bsdf = curVertex->getIntersection().getBSDF();
            pdfComponent = 1.f;

            // Hit emitter with null BSDF, break
            if (bsdf->getComponentCount() == 0) {
                return -2; // Invalid GP
            }

            // Select one of the BSDF component randomly
            Point2 randomBSDFSample = sampler->next2D();
            BSDFSamplingRecord bRec(curVertex->getIntersection(), sampler);
            int componentSelected = bsdf->sampleComponent(bRec, pdfComponent,
                                                          randomBSDFSample,
                                                          VertexClassifier::roughnessThreshold);

            // Check its roughness
            int componentQuery = (componentSelected == -1 ? 0 : componentSelected);
            bool roughFunction = bsdf->getRoughness(curVertex->getIntersection(),
                                                    componentQuery) > VertexClassifier::roughnessThreshold;
            //roughFunction &= !(curVertex->componentType & BSDF::EDelta);
            if (roughFunction) {
                // Take one component, if it is enough smooth,
                // Stop here and store the sampledComponentIndex
                // We will use this index later to decide the vertex classification
                curVertex->sampledComponentIndex = componentSelected;
                return t;
            }

            PathVertex *succVertex = pool.allocVertex();
            PathEdge *succEdge = pool.allocEdge();

            if (!curVertex->sampleNext(scene, sampler, predVertex, predEdge, succEdge,
                succVertex, ERadiance, rrStart != -1 && t >= rrStart, &throughput, componentSelected)) {
                pool.release(succVertex);
                pool.release(succEdge);
                return -1; // Invalid GP
            }
            
            append(succEdge, succVertex);

            predVertex = curVertex;
            curVertex = succVertex;
            predEdge = succEdge;
        }

        // If we arrive here, it's mean that we have reach the max bouncing
        const BSDF* bsdf = curVertex->getIntersection().getBSDF();
        pdfComponent = 1.f;

        // Hit emitter with null BSDF, break
        if(bsdf->getComponentCount() == 0) {
            return -2;
        }

        Point2 randomBSDFSample = sampler->next2D();
        BSDFSamplingRecord bRec(curVertex->getIntersection(), sampler);
        int componentSelected = bsdf->sampleComponent(bRec, pdfComponent,
                                                      randomBSDFSample,
                                                      VertexClassifier::roughnessThreshold);
        // Check its roughness
        int componentQuery = (componentSelected == -1 ? 0 : componentSelected);
        bool roughFunction = bsdf->getRoughness(curVertex->getIntersection(),
                                                componentQuery) > VertexClassifier::roughnessThreshold;
        //roughFunction &= !(curVertex->componentType & BSDF::EDelta);
        if (roughFunction) {
            // Take one component, if it is enough smooth,
            // Stop here and store the sampledComponentIndex
            // We will use this index later to decide the vertex classification
            curVertex->sampledComponentIndex = componentSelected;
            return t;
        } else {
            return -1;
        }
    }

};

class RadiusInitializer {
public:
    RadiusInitializer(const GPMConfig &config)
        : scene(0),
        m_maxDepth(-1), 
        m_rrDepth(-1), 
        m_config(config) 
    {

    }

    virtual ~RadiusInitializer() {
    }

    virtual void init(Scene *scene, int maxDepth, int rrDepth,
        std::vector<std::vector<GatherPoint>> &gatherPoints,
        const std::vector<Point2i> &offset) {
        SLog(EInfo, "Initialize GP generation object");

        // Copy the object
        // These will be used to generate
        // the gather points through the scene
        this->scene = scene;
        this->m_maxDepth = maxDepth;
        this->m_rrDepth = rrDepth;
        this->gatherBlocks = &gatherPoints;
        this->offset = &offset;

        // And then initialise the samplers
        // without shift in the random
        // sequence
        generateSamplers(0);

        m_pools.resize(offset.size());
    }

    void generateSamplers(int shiftRandomNumber) {
        // Initialize samplers for generate GP
        // Each pixel blocks will have is own
        // sampler to generate the same set
        // of gather points

        Properties props("independent");
        if (m_config.referenceMod) {
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

    virtual Float getRadiusRayDifferential(RayDifferential &ray, Float totalDist) {

        if (ray.hasDifferentials) {  // nbComponentExtra == 0 &&
            ray.scaleDifferential(1.f);  // Scale to one (TODO: Is it necessary ??? )
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
        }
        else {
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
        for (int i = 0; i < (int)gatherBlocks->size(); ++i) {  // For all gather points
            // Get the sampler associated
            // to the block
            ref<Sampler> sampler = samplers[i];
            MemoryPool& pool = m_pools[i];

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
                    
                    GatherPoint &gatherPoint = (gb[index++]);
                    gatherPoint.resetTemp(); // Reset M and collected flux associated to GP

                    // Initialize the GP plane position
                    gatherPoint.pos = Point2i(xofs + xofsInt, yofs + yofsInt);
                    sampler->generate(gatherPoint.pos);

                    // Invalid - for now
                    gatherPoint.depth = -1;
                    gatherPoint.pdf[EPdfSolidAngle] = gatherPoint.pdf[EPdfArea] = 0.0f;
                    gatherPoint.radius = 1;
                    gatherPoint.currEmission = Spectrum(0.f);
                    gatherPoint.pureSpecular = true;

                    // Trace path
                    if (gatherPoint.path == NULL) {
                        gatherPoint.path = new PathEx;      // Only allocate once because initialize will clear the memory pool automatically
                    }

                    gatherPoint.path->initialize(scene, 0.f, ERadiance, pool);
                    Float pdfComponent = 1.f;
                    
                    int maxVertexID = (m_maxDepth - 1) + 1;   // nSteps parameter
                    int lastVertexID = ((PathEx *)gatherPoint.path)->randomWalkFromPixelToFirstDiffuse(
                        scene, sampler, maxVertexID, gatherPoint.pos, m_rrDepth, pool, pdfComponent);
                    
                    for (int i = 2; i < gatherPoint.path->vertexCount() - 1; ++i) {
                        
                        bool diffuse = !gatherPoint.path->vertex(i)->degenerate;
                        if (m_config.nearSpecularDirectImage) {
                            // Relax the pureSpecular check by allowing high glossy path. 
                            // Note that this is a double-edged sword. It can leave some highlight regions as is (no reconstruction)
                            // if bounceRoughness is set too high.
                            diffuse = (VertexClassifier::type(*gatherPoint.path->vertex(i), gatherPoint.path->vertex(i)->sampledComponentIndex) == VERTEX_TYPE_DIFFUSE);
                        }
                        
                        if (diffuse) {
                            gatherPoint.pureSpecular = false;
                            break;
                        }
                    }

                    // Emitter hit: we assume that emitter always has a NullBSDF.  
                    // The tracing will stop when a NullBSDF is encountered.
                    if (lastVertexID == -2) {
                        gatherPoint.depth = -1;
                        
                        Path *pt = gatherPoint.path;
                        if(m_config.directTracing && pt->vertexCount() > 2) {
                            Spectrum weight = pt->vertex(0)->weight[ERadiance] *
                                              pt->vertex(0)->rrWeight *
                                              pt->edge(0)->weight[ERadiance];
                            for (int i = 1; i < (int)pt->vertexCount() - 1; i++) {
                                weight *= pt->vertex(i)->weight[ERadiance] *
                                            pt->vertex(i)->rrWeight *
                                            pt->edge(i)->weight[ERadiance];
                            }
                            // Complete the contribution with light emission
                            PathVertex *last = pt->vertex((int)pt->vertexCount() - 1);
                            if (last->getType() == PathVertex::ESurfaceInteraction) {
                                const Intersection& itsLocal = last->getIntersection();
                                if (itsLocal.isEmitter()) {
                                    int depth = pt->vertexCount() - 2;
                                    if (depth >= m_config.minDepth) {
                                        Spectrum emission = weight * itsLocal.Le(pt->edge(pt->vertexCount() - 2)->d);
                                        if (gatherPoint.pureSpecular == false) {
                                            gatherPoint.currEmission += emission;
                                            gatherPoint.emission += emission;
                                        }
                                        else {
                                            // Separate emission due to pure specular path
                                            // Usually, these causes bright edges and hence artifacts in L2 reconstruction
                                            // We treat them as direct image and add it after reconstruction.
                                            gatherPoint.directEmission += emission;
                                        }
                                    }
                                }
                            }

                            // If found GP pdf:
                            gatherPoint.pdf[EPdfSolidAngle] = 1.0f;
                            for (int i = 1; i < pt->vertexCount() - 1; ++i) {
                                gatherPoint.pdf[EPdfSolidAngle] *= pt->vertex(i)->pdf[ERadiance];
                                if (pt->vertex(i)->measure == EArea) {
                                    gatherPoint.pdf[EPdfSolidAngle] /= geometryOpposingTerm(pt->vertex(i)->getPosition(), pt->vertex(i + 1)->getPosition(), pt->vertex(i + 1)->getGeometricNormal());
                                }
                            }

                            // Pdf (area measure)
                            gatherPoint.pdf[EPdfArea] = 1.0f;
                            for (int i = 0; i < pt->vertexCount() - 1; ++i) {
                                gatherPoint.pdf[EPdfArea] *= pt->vertex(i)->pdf[ERadiance];
                            }
                        }

                        sampler->advance();
                        continue;
                    }

                    // Sanity check
                    PathVertex *last = gatherPoint.path->vertex((int)gatherPoint.path->vertexCount() - 1);
                    
                    // Strictly no glossy gather point
                    if (last->getType() != PathVertex::ESurfaceInteraction ||
                        VertexClassifier::type(*last, last->sampledComponentIndex) != VERTEX_TYPE_DIFFUSE
                    ) {

                        if(m_config.directTracing && last->getType() != PathVertex::ESurfaceInteraction) {
                            // If no intersection, check the envmap
                            //gatherPoint.emission += scene->evalEnvironment(ray); // No weight => bounce over scene
                        }
                        gatherPoint.depth = -1;

                    } else {

                        // Contribution of the path at the gather point
                        Path *pt = gatherPoint.path;
                        Spectrum weight = pt->vertex(0)->weight[ERadiance] *
                                          pt->vertex(0)->rrWeight *
                                          pt->edge(0)->weight[ERadiance];
                        for (int i = 1; i < (int)pt->vertexCount() - 1; i++) {
                                weight *= pt->vertex(i)->weight[ERadiance] *
                                          pt->vertex(i)->rrWeight *
                                          pt->edge(i)->weight[ERadiance];
                        }

                        // Pdf (solid angle measure) of the gather point
                        gatherPoint.pdf[EPdfSolidAngle] = 1.0f;
                        for (int i = 1; i < pt->vertexCount() - 1; ++i) {
                            gatherPoint.pdf[EPdfSolidAngle] *= pt->vertex(i)->pdf[ERadiance];
                            if (pt->vertex(i)->measure == EArea) {
                                gatherPoint.pdf[EPdfSolidAngle] /= geometryOpposingTerm(pt->vertex(i)->getPosition(), pt->vertex(i + 1)->getPosition(), pt->vertex(i + 1)->getGeometricNormal());
                            }
                        }

                        // Pdf (area measure) 
                        gatherPoint.pdf[EPdfArea] = 1.0f;
                        for (int i = 0; i < pt->vertexCount() - 1; ++i) {
                            gatherPoint.pdf[EPdfArea] *= pt->vertex(i)->pdf[ERadiance];
                        }

                        gatherPoint.depth = lastVertexID - 1;
                        gatherPoint.weight = weight;
                        gatherPoint.its = last->getIntersection();

                        // Sample the end point one more time to generate the BSDF component type 
                        // and component index at the gather point
                        int componentSelected = last->sampledComponentIndex;
                        gatherPoint.sampledComponent = componentSelected;
                        gatherPoint.pdfComponent = pdfComponent;
                        // pdf of the component will be accounted during gathering

                        // Check the PDF for sample an component
                        // There is a problem when it's equal to 0
                        // Arrives when ray leaking in RoughtPlastic
                        if (pdfComponent == 0) {
                            if (componentSelected != -1) {
                                SLog(EError, "Not the good component is returned for component selection");
                            }
                            break;
                        }

                        // Sanity check
                        // -1 for component selected = same behavior for all the bsdf component
                        // in this case, the probability to pick all need to be 1
                        if (componentSelected == -1 && pdfComponent != 1.f) {
                            SLog(EError, "All component is selected but the pdf is not 1");
                        }


                        // Compute the radius using the ray differential if this is the first gather point
                        // Note that this radius is rescale by the scale factor of the gather point
                        Point2 apertureSample;
                        Float timeSample = 0.0f;
                        if (needsApertureSample)
                            apertureSample = Point2(0.5f);
                        if (needsTimeSample)
                            timeSample = 0.5f;

                        
                        // Now estimate initial radius
                        const Point2 pixel = gatherPoint.path->vertex(1)->getSamplePosition();
                        RayDifferential rayCamera;
                        sensor->sampleRayDifferential(rayCamera, pixel, apertureSample, timeSample);

                        if (gatherPoint.scale == 0.f) {
                            SLog(EError, "Zero scale on valid gather point");
                        }

                        Float traveledDistance = 0.f; 
                        for (int i = 0; i < gatherPoint.path->vertexCount() - 1; ++i) {
                            traveledDistance += gatherPoint.path->edge(i)->length;
                        }
                        Float estimatedRadius = getRadiusRayDifferential(rayCamera, traveledDistance);
                        
                        // Should not be zero
                        estimatedRadius = std::max<Float>(estimatedRadius, m_config.epsilonRadius);

                        /*
                        // Use global initial radius
                        Float estimatedRadius = m_config.initialRadius;
                        */

                        // Radius of the current pass is the initial radius times the reduction ratio
                        gatherPoint.radius = estimatedRadius * gatherPoint.scale;

                        // Finish, we have generated a gather point
                    }

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
                        // No radius => Error because we will lose the flux
                        SLog(EError, "Valid GP with null radius");
                    }
                    else {
                        gp.rescaleRadii(gp.radius * gp.radius * M_PI);
                    }

                }
                else {
                    // No rescale, gp.flux is still radiance.
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
    int m_maxDepth;
    int m_rrDepth;

    // === Gather blocks attributes
    std::vector<std::vector<GatherPoint>> *gatherBlocks;
    const std::vector<Point2i> *offset;

    // === Sampler attributes
    ref_vector <Sampler> samplers;
    std::vector<MemoryPool> m_pools;

    const GPMConfig &m_config;
};

MTS_NAMESPACE_END
