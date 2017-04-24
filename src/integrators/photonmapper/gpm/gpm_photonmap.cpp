#include "gpm_photonmap.h"

#include <mitsuba/render/gatherproc.h>
#include <mitsuba/core/statistics.h>
#include <mitsuba/bidir/manifold.h>
#include <mitsuba/bidir/mut_manifold.h>

MTS_NAMESPACE_BEGIN

const Float D_EPSILON = (Float)(1e-14);

StatsCounter intersectionSucessRatio("GPM",
                                     "Interesected surface projection", EPercentage);
StatsCounter contributiveShiftedPRatio("GPM",
                                       "Have contribution photon", EPercentage);

StatsCounter MESuccess("GPM",
                       "Manifold Exp. sucessful shift", EPercentage);
StatsCounter MEMisSuccess("GPM",
                          "MIS computation succesfull", EPercentage);

StatsCounter MEShift("GPM",
                     "Percentage ME Shift    : ", EPercentage);
StatsCounter HVShift("GPM",
                     "Percentage HVCopy Shift: ", EPercentage);
StatsCounter DiffShift("GPM",
                     "Percentage Diff. Shift : ", EPercentage);
StatsCounter InvalidShift("GPM",
                      "Percentage Inval Shift : ", EPercentage);

/// Result of a half-vector duplication shift.
struct HalfVectorShiftResult {
    bool success;   ///< Whether the shift succeeded.
    Float jacobian; ///< Local Jacobian determinant of the shift.
    Vector3 wo;     ///< Tangent space outgoing vector for the shift.
};

/// Calculates the outgoing direction of a shift by duplicating the local half-vector.
static HalfVectorShiftResult halfVectorShift(Vector3 tangentSpaceMainWi, Vector3 tangentSpaceMainWo, Vector3 tangentSpaceShiftedWi, Float mainEta, Float shiftedEta) {
    HalfVectorShiftResult result;

    if (Frame::cosTheta(tangentSpaceMainWi) * Frame::cosTheta(tangentSpaceMainWo) < (Float)0) {
        // Refraction.

        // Refuse to shift if one of the Etas is exactly 1. This causes degenerate half-vectors.
        if (mainEta == (Float)1 || shiftedEta == (Float)1) {
            // This could be trivially handled as a special case if ever needed.
            result.success = false;
            return result;
        }

        // Get the non-normalized half vector.
        Vector3 tangentSpaceHalfVectorNonNormalizedMain;
        if (Frame::cosTheta(tangentSpaceMainWi) < (Float)0) {
            tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi * mainEta + tangentSpaceMainWo);
        }
        else {
            tangentSpaceHalfVectorNonNormalizedMain = -(tangentSpaceMainWi + tangentSpaceMainWo * mainEta);
        }

        // Get the normalized half vector.
        Vector3 tangentSpaceHalfVector = normalize(tangentSpaceHalfVectorNonNormalizedMain);

        // Refract to get the outgoing direction.
        Vector3 tangentSpaceShiftedWo = refract(tangentSpaceShiftedWi, tangentSpaceHalfVector, shiftedEta);

        // Refuse to shift between transmission and full internal reflection.
        // This shift would not be invertible: reflections always shift to other reflections.
        if (tangentSpaceShiftedWo.isZero()) {
            result.success = false;
            return result;
        }

        // Calculate the Jacobian.
        Vector3 tangentSpaceHalfVectorNonNormalizedShifted;
        if (Frame::cosTheta(tangentSpaceShiftedWi) < (Float)0) {
            tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi * shiftedEta + tangentSpaceShiftedWo);
        }
        else {
            tangentSpaceHalfVectorNonNormalizedShifted = -(tangentSpaceShiftedWi + tangentSpaceShiftedWo * shiftedEta);
        }

        Float hLengthSquared = tangentSpaceHalfVectorNonNormalizedShifted.lengthSquared() / (D_EPSILON + tangentSpaceHalfVectorNonNormalizedMain.lengthSquared());
        Float WoDotH = std::abs(dot(tangentSpaceMainWo, tangentSpaceHalfVector)) / (D_EPSILON + std::abs(dot(tangentSpaceShiftedWo, tangentSpaceHalfVector)));

        // Output results.
        result.success = true;
        result.wo = tangentSpaceShiftedWo;
        result.jacobian = hLengthSquared * WoDotH;
    }
    else {
        // Reflection.
        Vector3 tangentSpaceHalfVector = normalize(tangentSpaceMainWi + tangentSpaceMainWo);
        Vector3 tangentSpaceShiftedWo = reflect(tangentSpaceShiftedWi, tangentSpaceHalfVector);

        // FIXME: according to the paper, it must be the constant direction here, not the sampled direction. 
        // In this case it does not matter because the cosine is the same.
        Float WoDotH = dot(tangentSpaceShiftedWo, tangentSpaceHalfVector) / dot(tangentSpaceMainWo, tangentSpaceHalfVector);
        Float jacobian = std::abs(WoDotH);

        result.success = true;
        result.wo = tangentSpaceShiftedWo;
        result.jacobian = jacobian;
    }

    return result;
}

/// Returns whether point1 sees point2.
static bool testVisibility(const Scene* scene, const Point3& point1, const Point3& point2, Float time) {
    Ray shadowRay;
    shadowRay.setTime(time);
    shadowRay.setOrigin(point1);
    shadowRay.setDirection(point2 - point1);
    shadowRay.mint = Epsilon;
    shadowRay.maxt = (Float)1.0 - ShadowEpsilon;

    return !scene->rayIntersect(shadowRay);
}

static inline Float geometryTerm(const Point &p, const Vector &pn, const Point &q, const Vector &qn) {
    Vector pq = q - p;
    return absDot(pn, normalize(pq)) * absDot(qn, normalize(pq)) / (pq.lengthSquared());
}

static inline Float geometryTerm(const Path &path, int b, int c) {
    Point p = path.vertex(b)->getPosition();
    Point q = path.vertex(c)->getPosition();
    Vector pn = path.vertex(b)->getGeometricNormal();
    Vector qn = path.vertex(c)->getGeometricNormal();
    Vector pq = q - p;
    return absDot(pn, normalize(pq)) * absDot(qn, normalize(pq)) / (pq.lengthSquared());
}

static inline Float geometryOpposingTerm(const Point &p, const Point &q, const Vector &qn) {
  Vector pq = q - p;
  return absDot(qn, normalize(pq)) / (pq.lengthSquared());
  //return absDot(qn, normalize(pq)) / (D_EPSILON + pq.lengthSquared());
}

static inline Float geometryOpposingTerm(const Path &path, int b, int c) {
  Point p = path.vertex(b)->getPosition();
  Point q = path.vertex(c)->getPosition();
  Vector qn = path.vertex(c)->getGeometricNormal();
  Vector pq = q - p;
  return absDot(qn, normalize(pq)) / (pq.lengthSquared());
  //return absDot(qn, normalize(pq)) / (D_EPSILON + pq.lengthSquared());
}

static inline Float cosine(const Path &path, int b, int c) {
    Point p = path.vertex(b)->getPosition();
    Point q = path.vertex(c)->getPosition();
    Vector pn = path.vertex(b)->getGeometricNormal();
    return absDot(pn, normalize(q - p));
}

// Structure to store the shift information
struct ShiftRecord {
  Spectrum throughtput;
  Float pdf[2];
  Float jacobian[2];

  ShiftRecord() : throughtput(1.0f) {
      pdf[EPdfSolidAngle] = pdf[EPdfArea] = 0.0f;
      jacobian[EPdfSolidAngle] = jacobian[EPdfArea] = 1.0f;
  }
};

static bool diffuseReconnection(ShiftRecord& sRec, const Intersection& newIts,
                                const PathVertex* baseVertex,
                                const PathVertex* parentVertex) {

  Float pdfValue;
  if (parentVertex->getType() == PathVertex::ESurfaceInteraction) {
    const Intersection &parentIts = parentVertex->getIntersection();

    // parentIts could be on an emitter (even though type is surface interaction).
    // We allow this because photon tracing allows path to bounce on a light source surface.

    SAssert(parentIts.p == parentVertex->getPosition());

    Vector3 pWo = parentIts.toLocal(normalize(newIts.p - parentIts.p));
    Vector3 pWi = parentIts.wi;

    const BSDF* parentBSDF = parentIts.getBSDF();
    BSDFSamplingRecord bRec(parentIts,
                            pWi,
                            pWo,
                            EImportance);
    bRec.component = parentVertex->sampledComponentIndex;
    if (bRec.component >= parentBSDF->getComponentCount()) {
      SLog(EWarn, "Invalid component request %d", bRec.component);
    }
    EMeasure measure = ESolidAngle;

    // Evaluate the BRDF from the parent vertex to the offset vertex
    sRec.throughtput *= parentBSDF->eval(bRec, measure);
    pdfValue = parentBSDF->pdf(bRec, measure);
    pdfValue *= parentBSDF->pdfComponent(bRec);

    // Adjoint BSDF for shading normals (see vertex.cpp for example)
    Float wiDotGeoN = dot(parentIts.geoFrame.n, parentIts.toWorld(pWi));
    Float woDotGeoN = dot(parentIts.geoFrame.n, parentIts.toWorld(pWo));

    if((Frame::cosTheta(bRec.wo) * wiDotGeoN) == 0.f)
      return false;

    sRec.throughtput *= std::abs(
            (Frame::cosTheta(bRec.wi) * woDotGeoN) /
            (Frame::cosTheta(bRec.wo) * wiDotGeoN));
  } else {
    // Parent vertex samples an out-going direction from the light.
    // The position sampling on the light is handled by the emitter super node.
    SAssert(parentVertex->getType() == PathVertex::EEmitterSample);

    const AbstractEmitter *emitter = parentVertex->getAbstractEmitter();
    const PositionSamplingRecord &pRec = parentVertex->getPositionSamplingRecord();
    DirectionSamplingRecord dRec;
    dRec.d = normalize(newIts.p - parentVertex->getPosition());       // world
    dRec.measure = ESolidAngle;
    sRec.throughtput *= emitter->evalDirection(dRec, pRec);
    pdfValue = emitter->pdfDirection(dRec, pRec);
  }

  sRec.pdf[EPdfSolidAngle] = pdfValue;

  // always solid angle measure so convert
  sRec.pdf[EPdfArea] = pdfValue * geometryOpposingTerm(parentVertex->getPosition(), newIts.p, newIts.geoFrame.n);

  sRec.jacobian[EPdfSolidAngle] = 
      geometryOpposingTerm(parentVertex->getPosition(), newIts.p, newIts.geoFrame.n)
    / geometryOpposingTerm(parentVertex->getPosition(), baseVertex->getPosition(), baseVertex->getGeometricNormal());

  sRec.jacobian[EPdfArea] = 1.0f;

  // Normalize the contribution
  Float Gold = geometryOpposingTerm(parentVertex->getPosition(), baseVertex->getPosition(), baseVertex->getGeometricNormal());
  sRec.throughtput  *= Gold / parentVertex->pdf[EImportance];
  return true;
}

/// Result of a reconnection shift.
struct ReconnectionShiftResult {
    bool success;   ///< Whether the shift succeeded.
    Float jacobian; ///< Local Jacobian determinant of the shift.
    Vector3 wo;     ///< World space outgoing vector for the shift.
};

/// Tries to connect the offset path to a specific vertex of the main path.
static ReconnectionShiftResult reconnectShift(const Scene* scene, Point3 mainSourceVertex, Point3 targetVertex, Point3 shiftSourceVertex, Vector3 targetNormal, Float time) {
    ReconnectionShiftResult result;

    // Check visibility of the connection.
    Ray shadowRay(shiftSourceVertex,
                  targetVertex - shiftSourceVertex,
                  Epsilon, 1.0-ShadowEpsilon, time);
    if (scene->rayIntersect(shadowRay)) {
        // Since this is not a light sample, we cannot allow shifts through occlusion.
        result.success = false;
        return result;
    }

    // Calculate the Jacobian.
    Vector3 mainEdge = mainSourceVertex - targetVertex;
    Vector3 shiftedEdge = shiftSourceVertex - targetVertex;

    Float mainEdgeLengthSquared = mainEdge.lengthSquared();
    Float shiftedEdgeLengthSquared = shiftedEdge.lengthSquared();

    Vector3 shiftedWo = -shiftedEdge / math::safe_sqrt(shiftedEdgeLengthSquared);

    Float mainOpposingCosine = dot(mainEdge, targetNormal) / math::safe_sqrt(mainEdgeLengthSquared);
    Float shiftedOpposingCosine = dot(shiftedWo, targetNormal);

    Float jacobian = std::abs(shiftedOpposingCosine * mainEdgeLengthSquared) / (D_EPSILON + std::abs(mainOpposingCosine * shiftedEdgeLengthSquared));

    // Return the results.
    result.success = true;
    result.jacobian = jacobian;
    result.wo = shiftedWo;
    return result;
}

/// Describes the state of a ray that is being traced in the scene.
struct RayState {
    RayState() :
        eta(1.0f),
        throughput(Spectrum(0.0f)),
        alive(true)
    {
        pdf[EPdfSolidAngle] = pdf[EPdfArea] = 0.0f;
        jacobian[EPdfSolidAngle] = jacobian[EPdfArea] = 1.0f;
    }

    RayDifferential ray;             ///< Current ray.

    Spectrum throughput;             ///< Current throughput of the path.
    Float pdf[2];                       ///< PDF of the path. When the path is an offset path, this stores 
                                     /// the PDF as if the path is sampled with the same events as of the base path.
    Float jacobian[2];

    Intersection prevIts;
    Intersection its;

    Float eta;                       ///< Current refractive index of the ray.
                                     ///< For R.R use only.
    bool alive;                      ///< Whether the path matching to the ray is still good. Otherwise it's an invalid offset path with zero PDF and throughput.
};

static EVertexType vertexType(Intersection& its, RayDifferential& ray, int comp) {
  const BSDF* bsdf = its.getBSDF(ray);
  return VertexClassifier::type(bsdf, its, comp);
}

static inline Float cosineRatio(const Point &source, const Vector &sourceNormal, const Point &target, const Vector &targetNormal) {
    Vector edge = normalize(target - source);
    return std::abs(dot(edge, targetNormal) / dot(edge, sourceNormal));
}

static ELightShiftType getTypeShift(const Path *lt, int currVertex, int& b) {
  // Search next diffuse vertex
  b = -1;
  for (int i = currVertex - 1; i > 0 && b == -1; --i) {
    EVertexType vertexType = VertexClassifier::type(*lt->vertex(i), lt->vertex(i)->sampledComponentIndex);
    b = vertexType == VERTEX_TYPE_DIFFUSE ? i : -1;
  }

  if (b == -1) {
    // Error?
    return EInvalidShift;
  } else if (b + 1 == currVertex) {
    // Just use diffuse reconnection here
    return EDiffuseShift;
  } else {
    // If we are before the light source
    // it is possible to choose between
    // Manifold exploration or half vector copy
    if (b != 1) {
      EVertexType nextDiffuse = VertexClassifier::type(*lt->vertex(b - 1),
                                                       lt->vertex(b - 1)->sampledComponentIndex);
      if (nextDiffuse == VERTEX_TYPE_DIFFUSE) {
        return EHalfVectorShift;
      }
    }

    return EManifoldShift;
  }
}


bool GradientSamplingRecord::shiftPhoton(const Path *source, int currVertex, const GatherPoint *shiftGather,
                                         GradientSamplingResult &result) {
  if (shiftGather == NULL || shiftGather->depth != baseGather->depth) {
    // Invalid gather point
    return false;
  }

  // Maintain the photon distribution at the shift gather point to be the same as the base
  const PathVertex *basePhoton = source->vertex(currVertex);

  Point newVLoc;

  bool almostSameNormal = (shiftGather->its.shFrame.n - baseGather->its.shFrame.n).lengthSquared() < 0.01;
  if (almostSameNormal) {
    Vector localOffset = baseGather->its.toLocalCoherent(basePhoton->getPosition() - baseGather->its.p);
    newVLoc = shiftGather->its.p + shiftGather->its.toWorldCoherent(localOffset);
  }

  // Just test the visibility aka. projection from previous point
  // So compute the new direction of photon emission from the previous vertex
  const PathVertex* prevVertex = source->vertex(currVertex - 1);
  Vector dProj = normalize(newVLoc - prevVertex->getPosition());

  // Project the photon to the shifted photon
  Intersection itsProj;
  Ray projRay(prevVertex->getPosition(), dProj, 0.f);
  if (!scene->rayIntersect(projRay, itsProj)) {
    return false;
  }

  // Minor: Add a term to account for tracing event from parent photon
  // Inspired from Fig. 8 in HSLT paper
  // Due to tracing, dx'_k / dx_k is not exactly 1.
  int c = currVertex;
  Vector imgNormal = shiftGather->its.geoFrame.n;
  result.jacobian[EPdfArea] = geometryOpposingTerm(source->vertex(c - 1)->getPosition(), newVLoc, imgNormal)
                            / geometryOpposingTerm(source->vertex(c - 1)->getPosition(), itsProj.p, itsProj.geoFrame.n);
  if (result.jacobian[EPdfArea] == 0.0f) {
      // Sometimes the direction to the candidate offset photon is perpendicular to imgNormal, 
      // causing jacobian to be zero. 
      // We ignore such cases by resetting jacobian to 1, 
      // as eventually we only need the on-surface offset photon
      result.jacobian[EPdfArea] = 1.0f;
  }

  // Search next diffuse vertex
  int b;
  ELightShiftType type = getTypeShift(source, currVertex, b);
  RayDifferential projRayDiff(projRay);
  bool shiftedEnoughRough = VertexClassifier::enoughRough(itsProj, itsProj.getBSDF(projRayDiff));

  MEShift.incrementBase();
  HVShift.incrementBase();
  DiffShift.incrementBase();
  InvalidShift.incrementBase();

  if(type == EInvalidShift) {
    // Error?
    ++InvalidShift;
    return false;
  } else if(type == EDiffuseShift && shiftedEnoughRough) {
    // Just use diffuse reconnection here
    ++DiffShift;
    return shiftPhotonDiffuse(result, source, currVertex, itsProj, shiftGather);
  } else if(type == EHalfVectorShift ||
           (type == EDiffuseShift && !shiftedEnoughRough) ||
           (type == EManifoldShift)) {
      if (!config.useManifold) {
          ++InvalidShift;
          return false; // Impossible to handle this path
      } else {
          ++MEShift;
          return shiftPhotonManifold(result, source, currVertex, b, itsProj, shiftGather);
      }
  } else {
      SLog(EError, "Invalid type");
      ++InvalidShift;
      return false;
  }
}

static inline Matrix2x2 inverse(const Matrix2x2 &m) {
    Matrix2x2 target;
    m.invert(target);
    return target;
}

static inline void blockAssign(Matrix4x4 &m, int i, int j, const Matrix2x2 &b) {
    m(2 * i    , 2 * j    ) = b(0, 0);
    m(2 * i    , 2 * j + 1) = b(0, 1);
    m(2 * i + 1, 2 * j    ) = b(1, 0);
    m(2 * i + 1, 2 * j + 1) = b(1, 1);
}

bool GradientSamplingRecord::shiftPhotonManifold(GradientSamplingResult &result,
                                                 const Path *lt, int c, int b,
                                                 const Intersection& itsProj,
                                                 const GatherPoint *shiftGather) {


    // Perform a walk through a specular chain to a non-specular vertex
    const Path &source = (Path &)*lt;
    Path proposal;
    
    /* Allocate memory for the proposed path */
    proposal.clear();
    proposal.append(source, 0, b);		// Note that vertex b cannot be shared, because the pdf[EImportance] will be changed.
    for (int i = b; i <= c; ++i) {
        proposal.append(thdata.pool.allocEdge());            // outgoing edge at vertex b - 1
        memset(proposal.edge(proposal.edgeCount() - 1), 0, sizeof(PathEdge));  // make sure edges are ALWAYS initialized!

        proposal.append(thdata.pool.allocVertex());          // next vertex
        memset(proposal.vertex(proposal.vertexCount() - 1), 0, sizeof(PathVertex));
    }

    // Assign information for b 
    *proposal.edge(b - 1) = *source.edge(b-1);
    *proposal.vertex(b) = *source.vertex(b);

    // Assign information for c 
    proposal.vertex(c)->getIntersection() = itsProj;
    proposal.vertex(c)->type = PathVertex::ESurfaceInteraction;

    // Manifold walk between b .. c
    MESuccess.incrementBase();
    bool walkSuccess = thdata.offsetGenerator->manifoldWalkGPM(source, proposal, 1, // towards sensor
                                                        b, c);
    if (!walkSuccess) {
        for (int i = b; i <= c; ++i) {
            thdata.pool.release(proposal.edge(i - 1));
            thdata.pool.release(proposal.vertex(i));
        }
        result.weight = 1;
        return false;
    }
    ++MESuccess;

    // From GBDPT_proc.cpp: 
    // Copy missing component types, rrWeights, and sampledComponentsIndex from source path
	// This also makes sure that the shift is invertible (especially the later one, since we cannot force Mitsuba to always use the same component)
	for (int i = 0; i <= proposal.length(); i++) {
		proposal.vertex(i)->rrWeight = source.vertex(i)->rrWeight;
		proposal.vertex(i)->sampledComponentIndex = source.vertex(i)->sampledComponentIndex;
		if (proposal.vertex(i)->getType() == PathVertex::ESurfaceInteraction && proposal.vertex(i)->componentType == 0) //BUG? this looks weird! (look at the second condition...)
			proposal.vertex(i)->componentType = source.vertex(i)->componentType;
	}

    // One more constraint for reversibility
    // The parent vertex of the base photon must see the offset photon, and vice versa
    bool visible = testVisibility(scene, proposal.vertex(c - 1)->getPosition(), source.vertex(c)->getPosition(), 0);
    if (!visible) {
        for (int i = b; i <= c; ++i) {
            thdata.pool.release(proposal.edge(i - 1));
            thdata.pool.release(proposal.vertex(i));
        }
        result.weight = 1.0f;
        return false;
    }
   
    // Evaluate photon power 
    Spectrum photonWeight(1.0f);
    Float shiftedPdf[2];
    shiftedPdf[EPdfSolidAngle] = 1.0f;
    shiftedPdf[EPdfArea] = 1.0f;   // We skip the pdf of unchanged path segments in MIS
    for (int i = 0; i < b; ++i) {
        photonWeight *= proposal.vertex(i)->weight[EImportance] *
                        proposal.vertex(i)->rrWeight *
                        proposal.edge(i)->weight[EImportance];
    }

    for (int i = b; i < c; ++i) {
        EMeasure measure;
        if (i == 1) {
            const AbstractEmitter *emitter = proposal.vertex(i)->getAbstractEmitter();
            const PositionSamplingRecord &pRec = proposal.vertex(i)->getPositionSamplingRecord();
            DirectionSamplingRecord dRec;
            dRec.d = normalize(proposal.vertex(i + 1)->getPosition() - proposal.vertex(i)->getPosition());       // world
            dRec.measure = measure = ESolidAngle;
            photonWeight *= emitter->evalDirection(dRec, pRec);
            
            // 
            shiftedPdf[EPdfArea] *= emitter->pdfDirection(dRec, pRec);

        } else {
            Intersection &offsetIts = proposal.vertex(i)->getIntersection();
            const BSDF* shiftedBSDF = offsetIts.getBSDF();

            Vector3 wi = normalize(proposal.vertex(i - 1)->getPosition() - offsetIts.p);
            Vector3 wo = normalize(proposal.vertex(i + 1)->getPosition() - offsetIts.p);
            BSDFSamplingRecord bRec(offsetIts, offsetIts.toLocal(wi), offsetIts.toLocal(wo), EImportance);
            bRec.component = source.vertex(i)->sampledComponentIndex;
            measure = (source.vertex(i)->getComponentType() & BSDF::EDelta) ? EDiscrete : ESolidAngle;

            photonWeight *= shiftedBSDF->eval(bRec, measure);
            
            shiftedPdf[EPdfArea] *= shiftedBSDF->pdf(bRec, measure);
            shiftedPdf[EPdfArea] *= shiftedBSDF->pdfComponent(bRec);
            if (photonWeight.isZero() == false && shiftedBSDF->pdfComponent(bRec) == 0) {
                SLog(EWarn, "Invalid pdfComponent implementation");
                shiftedBSDF->pdfComponent(bRec);
            }

            // Adjoint BSDF for shading normals (see vertex.cpp for example)
            Float wiDotGeoN = dot(offsetIts.geoFrame.n, wi);
            Float woDotGeoN = dot(offsetIts.geoFrame.n, wo);

            if (Frame::cosTheta(bRec.wo) * wiDotGeoN == 0.0f) {
                // This case happens sometimes with glossy material
                result.weight = 1.0f;
                return false;
            }

            photonWeight *= std::abs(
                (Frame::cosTheta(bRec.wi) * woDotGeoN) /
                (Frame::cosTheta(bRec.wo) * wiDotGeoN));

        }

        photonWeight /= source.vertex(i)->pdf[EImportance];
        
        // Evaluate the specular chain by the integral in manifold domain
        if (source.vertex(i)->measure == EArea) {

            // Traditional area integral 
            photonWeight *= geometryOpposingTerm(proposal, i, i + 1);
            shiftedPdf[EPdfArea] *= geometryOpposingTerm(proposal, i, i + 1);
        }

    }

    // Evaluate Jacobian (area measure)
    Float jacobian = result.jacobian[EPdfArea];  // middle term | dy_s' / dy_s |

    // Jacobian in area integral in original space
    SpecularManifold *manifold = thdata.offsetGenerator->getSpecularManifold();

    // Jacobian computed using det
    jacobian *= manifold->det(proposal, b, c) / manifold->det(source, b, c);

    if (jacobian <= 0.0 || !std::isfinite(jacobian)) {
      SLog(EWarn, "Invalid jacobian %g %d %d", jacobian, b, c);
      for (int i = b; i <= c; ++i) {
          thdata.pool.release(proposal.edge(i - 1));
          thdata.pool.release(proposal.vertex(i));
      }
      result.weight = 1.0f;
      return false;
    }
    photonWeight *= jacobian;

    // Perform photon mapping
    GPhoton offsetPhoton(itsProj, -1, photonWeight, c - 1, GPhoton::getPrevComponentType(source.vertex(c - 1)));
    offsetPhoton.its.wi = offsetPhoton.its.toLocal(normalize(proposal.vertex(c - 1)->getPosition() - proposal.vertex(c)->getPosition()));
    SAssert(source.vertex(c-1)->getComponentType() == proposal.vertex(c-1)->getComponentType());
    std::tuple<bool, Spectrum> photonContribution = getPhotonContrib(offsetPhoton, shiftGather, baseGather->radius);
    if (!std::get<0>(photonContribution)) {
        // The projected photon doesn't have any contribution:
        // 1) Not inside the radius of the shifted gather point.
        // 2) Share not the same normal as the gather points.
        // And hence, the shift is not reversible.
        result.weight = 1.0f;
        for (int i = b; i <= c; ++i) {
            thdata.pool.release(proposal.edge(i - 1));
            thdata.pool.release(proposal.vertex(i));
        }
        return false;
    }

    // Default no MIS weight
    result.weight = 0.5f;
    result.shiftedFlux = std::get<1>(photonContribution) * shiftGather->weight;
    result.jacobian[EPdfArea] = jacobian;

    // Pdf check for offset path
    Float offsetPdf = shiftGather->pdf[EPdfArea] * shiftedPdf[EPdfArea];
    if (offsetPdf == Float(0)) {
        //SLog(EWarn, "Invalid offset path");
        // This path has zero density due to manifold walk. 
        // It cannot be sampled by particle tracing, and hence no reversible.
        result.weight = 1.0f;
        if(!result.shiftedFlux.isZero()) {
          SLog(EWarn, "0 PDF path but with a throughput: %s\n Set to 0", result.shiftedFlux.toString().c_str());
          result.shiftedFlux = Spectrum(0.f);
        }
        // When this assert fails, it could be due to pdfComponent is not correctly implemented.
    }

    if (config.useMIS) {
        MEMisSuccess.incrementBase();

        Float basePdf = baseGather->pdf[EPdfArea];
        // No need to consider pdf of the unchanged segments
        for (int i = b; i < c; ++i) {
            basePdf *= source.vertex(i)->pdf[EImportance];
        }

        Float offsetPdf = shiftGather->pdf[EPdfArea] * shiftedPdf[EPdfArea];
        Float allJacobian = shiftGather->jacobian[EPdfArea] * jacobian;

        if (basePdf == Float(0)) {
            SLog(EWarn, "Invalid base path. This case should not happen.");
            result.weight = 0.0f;
        }
        else if (offsetPdf == Float(0)) {
            //SLog(EWarn, "Invalid offset path");
            // This path has zero density due to manifold walk. 
            // It cannot be sampled by particle tracing, and hence no reversible.
            result.weight = 1.0f;
            if(!result.shiftedFlux.isZero()) {
              SLog(EWarn, "0 PDF path but with a throughput: %s\n Set to 0 (MIS)", result.shiftedFlux.toString().c_str());
              result.shiftedFlux = Spectrum(0.f);
            }
        }
        else {
            result.weight = 1.0f / (1.0f + offsetPdf * allJacobian / basePdf);
        }

    }
    
    // Clean up
    for (int i = b; i <= c; ++i) {
        thdata.pool.release(proposal.edge(i - 1));
        thdata.pool.release(proposal.vertex(i));
    }

    return true;
}

bool GradientSamplingRecord::shiftPhotonDiffuse(GradientSamplingResult &result,
                                                const Path *lt, int currVertex,
                                                const Intersection& itsProj,
                                                const GatherPoint *shiftGather)
{

  // Maintain the photon distribution at the shift gather point to be the same as the base
  const PathVertex *baseVertex = lt->vertex(currVertex);
  const PathVertex* parentVertex = lt->vertex(currVertex - 1);

  ShiftRecord sRec;
  diffuseReconnection(sRec, itsProj, baseVertex, parentVertex);
  if (sRec.pdf[EPdfSolidAngle] == Float(0)) {
      result.weight = 1.0f;
      return false;
  }

  // To be strict, we must re-evaluate the BSDF value at the parent
  // Compute the flux of parent photon
  Spectrum photonWeight = lt->vertex(0)->weight[EImportance] *
    //lt->vertex(0)->rrWeight *
    lt->edge(0)->weight[EImportance];

  for (int i = 1; i < currVertex - 1; ++i) {
    photonWeight *= lt->vertex(i)->weight[EImportance] *
      //lt->vertex(i)->rrWeight *
      lt->edge(i)->weight[EImportance];
  }

  // Jacobian for solid angle integral does not need to account for photon projection
  photonWeight *= sRec.throughtput * sRec.jacobian[EPdfSolidAngle];

  // Assign the first intersection, remember to assign the incoming direction for the offset photon
  GPhoton offsetPhoton(itsProj, -1, photonWeight, currVertex - 1,
                       GPhoton::getPrevComponentType( lt->vertex(currVertex - 1)));
  offsetPhoton.its.wi = offsetPhoton.its.toLocal(normalize(parentVertex->getPosition() - itsProj.p));

  // Evaluate BSDF at the photon
  std::tuple<bool, Spectrum> photonContribution = getPhotonContrib(offsetPhoton, shiftGather, baseGather->radius);

  contributiveShiftedPRatio.incrementBase(1);
  if (!std::get<0>(photonContribution)) {
    // The projected photon doesn't any contribution:
    // 1) Not inside the radius of the shifted gp.
    // 2) Share not the same normal as the gather points
    result.weight = 1.0f;
    return false;
  }
  ++contributiveShiftedPRatio;

  // Finish
  result.shiftedFlux = std::get<1>(photonContribution) * shiftGather->weight;
  result.weight = 0.5f;

  // Compute MIS if needed
  if (config.useMIS) {
    SAssert(currVertex-1 > 0);

    Float basePdf = baseGather->pdf[EPdfArea];
    basePdf *= lt->vertex(currVertex - 1)->pdf[EImportance];

    Float offsetPdf = shiftGather->pdf[EPdfArea] * sRec.pdf[EPdfArea];
    if (offsetPdf == Float(0) || basePdf == Float(0)) {
        //SLog(EWarn, "Invalid path");
        result.weight = 1.0f;
        return false;
    }

    Float allJacobian = shiftGather->jacobian[EPdfArea] * sRec.jacobian[EPdfArea];
    result.weight = 1.0f / (1.0f + offsetPdf * allJacobian / basePdf);
  }
  return true;
}

struct RadianceQueryGradient {
    RadianceQueryGradient(GradientSamplingRecord& _gRec, int minDepth, int maxDepth) :
        gRec(_gRec), minDepth(minDepth), maxDepth(maxDepth)
    {}

    inline void operator()(const GPhotonNodeKD &nodePhoton) {
        GPhoton photon = nodePhoton.getData().getPhoton();
        GatherPoint *baseGather = gRec.baseGather;

        int pathLength = baseGather->depth + photon.depth;
        if (pathLength < minDepth || pathLength > maxDepth) return;

        // Debug one shift by cancel it
        if(gRec.config.debugShift != -1) {
          int b;
          ELightShiftType currShift = getTypeShift(nodePhoton.getData().lightPath,
                                                   nodePhoton.getData().vertexId, b);
          if(gRec.config.debugShift != currShift) {
            return; // Do not compute the photon contribution
          }
        }

        // Compute photon contribution and shift photons
        std::tuple<bool, Spectrum> rBase = gRec.getPhotonContrib(photon, baseGather);
        if (std::get<0>(rBase) == false) {
            // Not a valid base path
            return;
        }

        Spectrum curBaseFlux = baseGather->weight * std::get<1>(rBase);
        gRec.baseFlux += curBaseFlux;

        // Shift 
        const Point2 basePixel = baseGather->path->vertex(1)->getSamplePosition();
        const Point2 rightPixel = basePixel + Point2(1, 0);
        const Point2 leftPixel = basePixel + Point2(-1, 0);
        const Point2 bottomPixel = basePixel + Point2(0, -1);
        const Point2 topPixel = basePixel + Point2(0, 1);

        // Order must match those defined in EPixel
        const Point2 pixels[] = { leftPixel, rightPixel, topPixel, bottomPixel };

        Vector2i filmSize = gRec.scene->getFilm()->getSize();

        for (int i = 0; i < 4; ++i) {
            GradientSamplingResult result;

            // If the gather point are not traced
            // Generate them
            if(!gRec.shiftGPInitialized) {
              gRec.validShiftGP[i] = gRec.shiftGatherPoint(pixels[i], gRec.shiftGather[i]);
            }

            if(gRec.validShiftGP[i]) {
              gRec.shiftPhoton(nodePhoton.getData().lightPath,
                               nodePhoton.getData().vertexId,
                               &gRec.shiftGather[i], result);
            }
            
            // no reverse shift at right and top corners
            if ((i == ERight && (int)basePixel.x == filmSize.x - 1) ||
                (i == ETop && (int)basePixel.y == filmSize.y - 1))
            {
                result.weight = 1.0f;
            }

            // Weight checking
            if(result.weight > 1.0f || result.weight < 0.f) {
                SLog(EError, "Weight invalid: %f", result.weight);
            }

            gRec.shiftedFlux[i] += result.weight * result.shiftedFlux;
            gRec.weightedBaseFlux[i] += result.weight * curBaseFlux;
        }

        gRec.shiftGPInitialized = true;
    }

    GradientSamplingRecord& gRec;
    int minDepth;
    int maxDepth;
};

size_t GPhotonMap::estimateRadianceGrad(GradientSamplingRecord& gRec, int minDepth, int maxDepth) const {
    RadianceQueryGradient query(gRec, minDepth, maxDepth);
    size_t count = m_kdtree.executeQuery(gRec.baseGather->its.p, gRec.baseGather->radius, query);
    return count;
}

bool GradientSamplingRecord::shiftGatherPoint(const Point2 offsetPixel, GatherPoint &shiftGather,
    bool emissionShift) {

    GatherPoint *baseGather = this->baseGather;
    Path *basePath = baseGather->path;
    shiftGather.clear();

    // Find primary hit point
    Sensor *sensor = scene->getSensor();
    Point2 apertureSample;
    Float timeSample = 0.0f;
    if (sensor->needsApertureSample())
        apertureSample = Point2(0.5f);
    if (sensor->needsTimeSample())
        timeSample = 0.5f;

    // Sample the primary ray from the camera
    Ray ray;
    sensor->sampleRay(ray, offsetPixel, apertureSample, timeSample);

    shiftGather.weight = Spectrum(1.0f);   // Solid angle case

    shiftGather.jacobian[EPdfSolidAngle] = shiftGather.jacobian[EPdfArea] = 1.0f;
    shiftGather.pdf[EPdfSolidAngle] = shiftGather.pdf[EPdfArea] = 1.0f;

    // SH: experimental ///////////////////////////
    shiftGather.jacobian[EPdfArea] = 1.0f;

    // Area measure of Jacobian of the first hit point
    // We use pdf ratio to calculate the Jacobian term
    Float pdf1, pdf2;

    Point org = basePath->vertex(1)->getPosition();
    {
        Point first = basePath->vertex(2)->getPosition(); // first hit point;
        Vector firstN = basePath->vertex(2)->getGeometricNormal();
        PositionSamplingRecord pRecSrc = basePath->vertex(1)->getPositionSamplingRecord();
        DirectionSamplingRecord dRecSrc;
        dRecSrc.d = normalize(first - org);
        dRecSrc.measure = ESolidAngle;
        pdf1 = sensor->pdfDirection(dRecSrc, pRecSrc);
    }

    {
        scene->rayIntersect(ray, shiftGather.its);
        if (!shiftGather.its.isValid()) {
            return false;
        }
        PositionSamplingRecord pRecDst;
        pRecDst.measure = EArea;
        pRecDst.time = 0.0f;
        DirectionSamplingRecord dRecDst;
        dRecDst.d = normalize(shiftGather.its.p - ray.o);
        dRecDst.measure = ESolidAngle;
        pdf2 = sensor->pdfDirection(dRecDst, pRecDst);

        shiftGather.pdf[EPdfSolidAngle] = pdf2;
        shiftGather.pdf[EPdfArea] = pdf2;
    }

    shiftGather.jacobian[EPdfArea] *= pdf1 / pdf2;

    // If we use area measure, the contribution starts with this value.
    // So when it is multiplied with Jacobian, it goes back to 1.
    // This formula assumes that the ray is sampled according to the importance distribution.
    shiftGather.weight = Spectrum(pdf2 / pdf1);

    // At boundary, pdf of the shifted point is zero. 
    // We assume the Jacobian and weight to be 1 in such cases. 
    if (pdf2 == 0.0f) {
        shiftGather.jacobian[EPdfArea] = 1.0f;
        shiftGather.weight = Spectrum(1.0f);
        shiftGather.pdf[EPdfSolidAngle] = pdf1;
        shiftGather.pdf[EPdfArea] = pdf1;
    }


    ///////////////////////////////


    EMeasure measure = ESolidAngle;   // Remember to convert the pdf and Jacobian for the first vertex

    for (int i = 2; i < basePath->vertexCount(); ++i) {
        PathVertex *baseVertex = basePath->vertex(i);

        scene->rayIntersect(ray, shiftGather.its);
        if (!shiftGather.its.isValid()) {
            return false;
        }

        // Manage area pdf in previous round
        if (measure == ESolidAngle) {
            shiftGather.pdf[EPdfArea] *= geometryOpposingTerm(ray.o, shiftGather.its.p, shiftGather.its.geoFrame.n);

            shiftGather.jacobian[EPdfArea] *=
                geometryOpposingTerm(basePath->vertex(i - 1)->getPosition(), basePath->vertex(i)->getPosition(), basePath->vertex(i)->getGeometricNormal()) /
                geometryOpposingTerm(ray.o, shiftGather.its.p, shiftGather.its.geoFrame.n);

            // Evaluate weight in area measure
            shiftGather.weight *= geometryOpposingTerm(ray.o, shiftGather.its.p, shiftGather.its.geoFrame.n) /
                geometryOpposingTerm(basePath->vertex(i - 1)->getPosition(), basePath->vertex(i)->getPosition(), basePath->vertex(i)->getGeometricNormal());
        }

        const BSDF* shiftedBSDF = shiftGather.its.getBSDF();

        if (!(emissionShift && i == basePath->vertexCount() - 1))  // This check is not applicable when shifting the last vertex on a complete path
        {
            // Deny shifts between Dirac and non-Dirac BSDFs
            bool bothDelta, bothSmooth;
            if (baseVertex->componentType != 0) {
                bothDelta = (baseVertex->componentType & BSDF::EDelta) && (shiftedBSDF->getType() & BSDF::EDelta);
                bothSmooth = (baseVertex->componentType & BSDF::ESmooth) && (shiftedBSDF->getType() & BSDF::ESmooth);
            }
            else {
                bothDelta = (baseVertex->getIntersection().getBSDF()->getType() & BSDF::EDelta) && (shiftedBSDF->getType() & BSDF::EDelta);
                bothSmooth = (baseVertex->getIntersection().getBSDF()->getType() & BSDF::ESmooth) && (shiftedBSDF->getType() & BSDF::ESmooth);
            }
            if (!(bothDelta || bothSmooth)) {
                return false;
            }
        }

        if (shiftedBSDF->getType() != baseVertex->getIntersection().getBSDF()->getType()) return false;


        // If we reach the last vertex, succeed.
        if (i == (int)basePath->vertexCount() - 1) break;

        // Perform half vector copy

        PathVertex *predVertex = basePath->vertex(i - 1);
        PathVertex *succVertex = basePath->vertex(i + 1);

        const Intersection &baseIts = baseVertex->getIntersection();
        Vector baseWi = normalize(predVertex->getPosition() - baseVertex->getPosition());
        Vector baseWo = normalize(succVertex->getPosition() - baseVertex->getPosition());

        Vector3 shiftWi;
        HalfVectorShiftResult shiftResult;

        // When the base and shifted gather point belong to two different triangles,
        // we still expect their tangent vectors to be similar 
        // to avoid artifact to appear at the triangle boundary.
        // Only apply this for smooth surface case. 
        bool almostSameNormal = (shiftGather.its.shFrame.n - baseIts.shFrame.n).lengthSquared() < 0.01;
        if (almostSameNormal) {
            shiftWi = shiftGather.its.toLocalCoherent(-ray.d);
            shiftResult = halfVectorShift(baseIts.toLocalCoherent(baseWi),
                baseIts.toLocalCoherent(baseWo),
                shiftWi,
                baseIts.getBSDF()->getEta(), shiftedBSDF->getEta());
        }
        
        
        if (baseVertex->componentType & BSDF::EDelta) {
            // Dirac delta integral is a point evaluation - no Jacobian determinant!
            shiftResult.jacobian = Float(1);
        }

        // Pre-multiply Jacobian
        if (!shiftResult.success) {
            // The shift is non-invertible so kill it.
            return false;
        }
        // Invertible shift, success.       
        shiftGather.jacobian[EPdfSolidAngle] *= shiftResult.jacobian;
        Vector shiftWo = shiftResult.wo;

        // Evaluate BSDF at the new vertex
        BSDFSamplingRecord bRec(shiftGather.its, shiftWi, shiftWo);
        bRec.component = baseVertex->sampledComponentIndex;
        measure = (baseVertex->getComponentType() & BSDF::EDelta) ? EDiscrete : ESolidAngle;

        if (bRec.component >= shiftedBSDF->getComponentCount()) {
            SLog(EWarn, "Invalid component request %d", bRec.component);
        }

        // For throughput evaluation, we just use solid angle measure. 
        // The value will be the same as area measure.
        shiftGather.weight *= shiftedBSDF->eval(bRec, measure);

        Float shiftPdfWo = shiftedBSDF->pdf(bRec, measure);
        shiftPdfWo *= shiftedBSDF->pdfComponent(bRec);
        shiftGather.pdf[EPdfSolidAngle] *= shiftPdfWo;

        // Further cache area pdf and jacobian for MIS use
        shiftGather.pdf[EPdfArea] *= shiftPdfWo;

        if (shiftGather.pdf[EPdfSolidAngle] == Float(0)) {
            // Offset path is invalid!
            return false;
        }

        // Account for the probability of the next vertex on base path
        // Turn existing area density into solid angle
        // Pdf of the component already accounted in pdf[2] array.
        Float baseWoPdf;
        if (baseVertex->measure != EDiscrete) {
            baseWoPdf = baseVertex->pdf[ERadiance] / geometryOpposingTerm(baseVertex->getPosition(),
                succVertex->getPosition(), succVertex->getGeometricNormal());
        }
        else {
            baseWoPdf = baseVertex->pdf[ERadiance];
        }

        // evalPdf does not take into account of the sampled component at the vertex
        //Float baseWoPdf = baseVertex->evalPdf(scene, predVertex, succVertex, ERadiance, measure);
        shiftGather.weight /= baseWoPdf;

        // Prepare for next tracing step
        Vector3 outgoingDirection;
        if (almostSameNormal)
            outgoingDirection = shiftGather.its.toWorldCoherent(shiftWo);

        // Strict normals check to produce the same results as bidirectional methods when normal mapping is used.
        if (config.strictNormals && dot(outgoingDirection, shiftGather.its.geoFrame.n) * Frame::cosTheta(bRec.wo) <= 0) {
            return false;
        }

        ray = Ray(shiftGather.its.p, outgoingDirection, ray.time);
    }

    // By default, weight is premultiplied with area measure Jacobian
    shiftGather.weight *= shiftGather.jacobian[EPdfArea];

    shiftGather.radius = baseGather->radius;
    shiftGather.pos = Point2i((int)offsetPixel.x, (int)offsetPixel.y);
    shiftGather.sampledComponent = baseGather->sampledComponent;
    shiftGather.pdfComponent = baseGather->pdfComponent;
    shiftGather.depth = (int)basePath->vertexCount() - 2;

    if (emissionShift) {
        if (shiftGather.its.isEmitter()) {
            if (shiftGather.depth >= config.minDepth) {
                if (baseGather->pureSpecular == false) {
                    shiftGather.emission += shiftGather.weight * shiftGather.its.Le(shiftGather.its.toWorld(shiftGather.its.wi));
                    shiftGather.currEmission += shiftGather.weight * shiftGather.its.Le(shiftGather.its.toWorld(shiftGather.its.wi));
                }
            }
        }
    }

  return true;
}

MTS_IMPLEMENT_CLASS_S(GPhotonMap, false, SerializableObject)
MTS_NAMESPACE_END