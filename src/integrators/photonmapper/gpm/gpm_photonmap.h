#pragma once

#include <mitsuba/core/kdtree.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/bidir/path.h>
#include <mitsuba/render/gatherproc.h>

#include "gpm_struct.h"

#include <list>
#include <tuple>

MTS_NAMESPACE_BEGIN

// === Useful structures
// for representing the photons or light path
class GPhoton {
public:
    Intersection its;
    int sampledComponent;

    Spectrum weight;
    int depth;

    int prevComponentType;       // default is zero, which means accepting all components

    GPhoton(const Intersection& _its, int _sampledComp,
        const Spectrum _w, int _d, int prevComponentType = 0) :
        its(_its), weight(_w), depth(_d), sampledComponent(_sampledComp),
        prevComponentType(prevComponentType)
    {}

    static int getPrevComponentType(const PathVertex *prev) {
        if (prev->isEmitterSample()) {
            // As emitter component type is 0, treat it as diffuse
            return BSDF::EDiffuseReflection;
        }
        else {
            return prev->componentType;
        }
    }
};

struct GPhotonNodeData {
    int vertexId;
    Path* lightPath;
    Spectrum weight;

    GPhotonNodeData() :
        vertexId(-1), lightPath(0), weight(0.f)
    {}

    GPhotonNodeData(Path* _lt, int _vID,
        Spectrum _weight) :
        vertexId(_vID), lightPath(_lt), weight(_weight)
    {}

    //TODO: Optimize this !!!
    GPhoton getPhoton() const {
        int vID = vertexId;
        const Path* lt = lightPath;
        const PathVertex* v = lt->vertex(vID);
        if (v->type != PathVertex::ESurfaceInteraction) {
            SLog(EError, "Bad vertex type !");
        }

        // Record component type for interaction mode selection
        const PathVertex *prev = lt->vertex(vID - 1);

        // For now just build a photon (not efficient but easy !)
        // depth is vertex ID - 1 because the path has a super node. 
        GPhoton photon(v->getIntersection(), v->sampledComponentIndex, weight, vID - 1, GPhoton::getPrevComponentType(prev));
        return photon;
    }
};

struct GradientSamplingResult {
    Spectrum shiftedFlux;           // contribution of the shifted path (weighted by density and Jacobian)
    Float weight;                   // MIS weight
    Float jacobian[2];              // Jacobian
    GradientSamplingResult() : shiftedFlux(0.0f), weight(1.0f) {
        jacobian[EPdfArea] = 1.0f;
        jacobian[EPdfSolidAngle] = 1.0f;
    }
};

// These structure is to store all the
// gather point information usefull for the gradient computation
// + Store the results of the photon map.
struct GradientSamplingRecord {
    GatherPoint* baseGather;
    Scene* scene;

    Spectrum baseFlux;                      // map to total flux defined in GatherPoint struct
    Spectrum shiftedFlux[4];            
    Spectrum weightedBaseFlux[4];

    GatherPoint shiftGather[4];
    bool validShiftGP[4];
    bool shiftGPInitialized;

    const GPMConfig& config;
    GPMThreadData& thdata;

    GradientSamplingRecord(Scene* _sc, GatherPoint* _c, const GPMConfig& _config,
      GPMThreadData& _thdata) :
        baseGather(_c),
        scene(_sc), baseFlux(0.f),
        config(_config), thdata(_thdata)
    {
        for (int i = 0; i < 4; ++i) {
            shiftedFlux[i] = Spectrum(0.0f);
            weightedBaseFlux[i] = Spectrum(0.0f);
            validShiftGP[i] = false;
        }
        shiftGPInitialized = false;
    }

    bool shiftGatherPoint(const Point2 offsetPixel, GatherPoint &shiftGather, bool emissionShift = false);

    bool shiftPhoton(const Path *lt, int currVertex, const GatherPoint *shiftGather,
                     GradientSamplingResult &result);
    bool shiftPhotonDiffuse(GradientSamplingResult &result,
                            const Path *lt, int currVertex,
                            const Intersection& itsProj,
                            const GatherPoint *shiftGather);
    bool shiftPhotonManifold(GradientSamplingResult &result,
                             const Path *lt, int c, int b,
                             const Intersection& itsProj,
                             const GatherPoint *shiftGather);

    void estimateGradEmitter() {
        // Shift
        const Point2 basePixel = baseGather->path->vertex(1)->getSamplePosition();
        const Point2 rightPixel = basePixel + Point2(1, 0);
        const Point2 leftPixel = basePixel + Point2(-1, 0);
        const Point2 bottomPixel = basePixel + Point2(0, -1);
        const Point2 topPixel = basePixel + Point2(0, 1);

        const Point2 pixels[] = { leftPixel, rightPixel, topPixel, bottomPixel };

        Vector2i filmSize = scene->getFilm()->getSize();

        for (int i = 0; i < 4; ++i) {
            Float miWeight = 1.f;
            if (shiftGatherPoint(pixels[i], shiftGather[i], true)) {
                if (shiftGather[i].its.isEmitter()) {
                    miWeight = 1.f / (1.f +
                      (shiftGather[i].pdf[EPdfArea] / baseGather->pdf[EPdfArea]) * shiftGather[i].jacobian[EPdfArea]);
                    if(miWeight > 1.f) {
                        SLog(EError, "MIS problem");
                    }

                    if ((i == ERight && (int)basePixel.x == filmSize.x - 1) ||
                      (i == ETop && (int)basePixel.y == filmSize.y - 1))
                    {
                        miWeight = 1.0f;
                    }

                    baseGather->shiftedEmitterFlux[i] += shiftGather[i].currEmission * miWeight;
                }
            } // Shift success

            baseGather->weightedEmitterFlux[i] += baseGather->currEmission * miWeight;
        }

    }

    std::tuple<bool, Spectrum> getPhotonContrib(const GPhoton& photon,
        const GatherPoint* gp, Float otherRadii = -1)
    {
        if (config.interactionMode != BSDF::EAll &&
            photon.prevComponentType > 0 && !(photon.prevComponentType & config.interactionMode)) {
            return std::make_tuple(false, Spectrum(0.f));
        }

        if (config.eyePathMaxDepth != -1 && gp->depth > config.eyePathMaxDepth) {
            return std::make_tuple(false, Spectrum(0.f));
        }

        if (config.eyePathMinDepth > 0 && gp->depth < config.eyePathMinDepth) {
            return std::make_tuple(false, Spectrum(0.f));
        }

        // Test the radius
        // if an other radii is provided, use this one
        Float lengthSqr = (gp->its.p - photon.its.p).lengthSquared();
        if (otherRadii > 0) {
            if ((otherRadii * otherRadii - lengthSqr) < 0)
                return std::make_tuple(false, Spectrum(0.f));
        }
        else {
            // No other radii is provided, just use GP radii
            if ((gp->radius * gp->radius - lengthSqr) < 0)
                return std::make_tuple(false, Spectrum(0.f));
        }

        Vector photonWi = photon.its.toWorld(photon.its.wi);
        Normal photonNormal = photon.its.geoFrame.n;

#ifndef MTS_NOSHADINGNORMAL
        Float wiDotGeoN = absDot(photonNormal, photonWi);
#endif
        
        
        if (dot(photonNormal, gp->its.shFrame.n) < 1e-1f
#ifndef MTS_NOSHADINGNORMAL
            || wiDotGeoN < 1e-2f
#endif
            )
        {
            return std::make_tuple(false, Spectrum(0.f));
        }

        // Accumulate the contribution of the photon
        BSDFSamplingRecord bRec(gp->its, gp->its.toLocal(photonWi), gp->its.wi, EImportance);
        bRec.component = gp->sampledComponent;

#ifdef MTS_NOSHADINGNORMAL
        Spectrum value = photon.weight * (gp->its.getBSDF()->eval(bRec)) / std::abs(Frame::cosTheta(bRec.wo));
#else
        Spectrum value = photon.weight * gp->its.getBSDF()->eval(bRec);
#endif

        value /= gp->pdfComponent;

        if (value.isZero()) {
            // Return true because this is still a sampleable path by photon mapping
            return std::make_tuple(true, Spectrum(0.f));
        }
        
        // Account for non-symmetry due to shading normals
        // In photon mapping case, woDotGeoN term is cancelled because photon gathering
        // does not involve a cosine term. 
#ifndef MTS_NOSHADINGNORMAL
        value *= std::abs(Frame::cosTheta(bRec.wi) / (wiDotGeoN * Frame::cosTheta(bRec.wo)));
#endif
        
        // Accumulate the results
        return std::make_tuple(true, value);
    }
};

// === Acceleration structure
// We need to wrap the photon information inside the Kd-tree node
class GPhotonNodeKD : public SimpleKDNode<Point, GPhotonNodeData> {
public:
    GPhotonNodeKD() {
    }

    /// Construct from a photon interaction
    GPhotonNodeKD(GPhotonNodeData _data) :
        SimpleKDNode<Point, GPhotonNodeData>(_data) {

        // Setup the position of the current node
        // using the lightpath informations
        this->setPosition(this->getData().getPhoton().its.p);
    }

    /// Unserialize from a binary data stream
    GPhotonNodeKD(Stream *stream) {
        SLog(EError, "No serialized possible ...");
    }
};



class GPhotonMap : public SerializableObject  {
public:
    //////////////////////////////////////////////////////////////////////////
    // Public type definition
    //////////////////////////////////////////////////////////////////////////
    typedef PointKDTree<GPhotonNodeKD >        PhotonTree;
    typedef PointKDTree<GPhotonNodeKD >::SearchResult   SearchResult;

public:
    GPhotonMap() {
        SLog(EError, "No empty constructor");
    }

    GPhotonMap(int numPhotons, int minDepth) :
        m_kdtree(0, PhotonTree::ESlidingMidpoint), m_minDepth(minDepth)
    {
        m_kdtree.reserve(numPhotons);

        m_photons.reserve(numPhotons);
    }

    GPhotonMap(Stream *stream, InstanceManager *manager) {
        SLog(EError, "Not allowed to serialize GPhotonMap");
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        SLog(EError, "Not allowed to serialize GPhotonMap");
    }

    template<typename Functor>
    void executeQuery(const Point& p, Functor& func, Float radii) const {
        m_kdtree.executeQuery(p, radii, func);
    }

    inline size_t nnSearch(const Point &p, size_t k,
        SearchResult *results) const {
        return m_kdtree.nnSearch(p, k, results);
    }

    size_t estimateRadianceGrad(GradientSamplingRecord& gRec, int minDepth, int maxDepth) const;

    size_t getDepth() const{ return m_kdtree.getDepth(); }

    inline GPhotonNodeKD* operator[](size_t idx) { return &m_kdtree[idx]; }

    inline size_t size() const { return m_kdtree.size(); }

    inline size_t capacity() const { return m_kdtree.capacity(); }

    inline int tryAppend(int idWorker, const Path *lightPath) {
        size_t nAppended = 0;

        // Check if the photon map is full
        if (size() < capacity()) {
            int startIndex = 2;
            if (lightPath->vertexCount() <= startIndex) {
                return 0;
            }

            // Add the light path
            m_lightpaths.push_back(std::make_tuple(idWorker, const_cast<Path*>(lightPath)));

            // Treat each vertex of the path as a photon
            Spectrum importanceWeights = lightPath->vertex(0)->weight[EImportance] *
                lightPath->vertex(0)->rrWeight *
                lightPath->edge(0)->weight[EImportance];

            // At least start from 2 to avoid super node and light node
            for (int i = startIndex; i < lightPath->vertexCount(); i++) {
                if (size() < capacity()) {
                    // Photon flux
                    importanceWeights *= lightPath->vertex(i - 1)->weight[EImportance] *
                      lightPath->vertex(i - 1)->rrWeight *
                      lightPath->edge(i - 1)->weight[EImportance];

                    // To be consistent with particleproc.cpp, we should only consider non-specular vertices
                    const BSDF *bsdf = lightPath->vertex(i)->getIntersection().getBSDF();
                    if (bsdf) {
                        int bsdfType = bsdf->getType();
                        if (!(bsdfType & BSDF::EDiffuseReflection) && !(bsdfType & BSDF::EGlossyReflection))
                            continue;
                    }

                    // Try to add the photon associated to the light path
                    m_kdtree.push_back(GPhotonNodeKD(
                      GPhotonNodeData(std::get<1>(m_lightpaths.back()), i, importanceWeights)));
                    m_photons.push_back(GPhotonNodeData(std::get<1>(m_lightpaths.back()), i, importanceWeights));
                    nAppended++;
                } else {
                    if (nAppended == 0) {
                        m_lightpaths.pop_back();
                    }
                    return nAppended;
                }
            }

            if (nAppended == 0) {
                m_lightpaths.pop_back();
            }
            return nAppended;

        }
        else {
            return -1;
        }
    }

    inline void build(bool recomputeAABB = false) { m_kdtree.build(recomputeAABB); }

    inline void clean(std::vector<MemoryPool>& pools) {
        // Clean memory
        for (auto p : m_lightpaths) {
            MemoryPool& pool = pools[std::get<0>(p)];
            std::get<1>(p)->release(pool);
            delete std::get<1>(p); // Free object memory
        }
        m_lightpaths.clear();
    }

    GPhoton getPhoton(int i) const {
        return m_photons[i].getPhoton();
    }

    const GPhotonNodeData &getPhotonNodeData(int i) const {
        return m_photons[i];
    }

    MTS_DECLARE_CLASS()
protected:
    PhotonTree m_kdtree;
    std::list<std::tuple<int, Path*>> m_lightpaths;
    int m_minDepth;

    // Experimental: store all photons in a linear array for light tracing path shift
    std::vector<GPhotonNodeData> m_photons;
};

MTS_NAMESPACE_END
