#pragma once

#include <mitsuba/render/gatherproc.h>
#include <mitsuba/bidir/mut_manifold.h>
#include <mitsuba/core/plugin.h>

MTS_NAMESPACE_BEGIN

/// Please don't be confused with ESolidAngle and EArea. 
enum EPdfMeasure {
    EPdfSolidAngle = 0,
    EPdfArea = 1
};

enum ELightShiftType {
  EDiffuseShift = 1<<1,
  EManifoldShift = 1<<2,
  EHalfVectorShift = 1<<3,
  EInvalidShift = 1<<4
};

/// Configuration for the gradient photon mapper.
struct GPMConfig {
    // Function to get the interaction mode from a string
    inline int getInteractionMode(const std::string& mode) {
      std::string lowMode = mode;
      std::transform(lowMode.begin(), lowMode.end(), lowMode.begin(), ::tolower);

      if(lowMode == "all") {
        return BSDF::EAll;
      } else if(lowMode == "diffuse") {
          return BSDF::EDiffuse;
      } else if (lowMode == "specular") {
          return BSDF::EDelta;
      } else if (lowMode == "glossy") {
          return BSDF::EGlossy;
      } else {
        SLog(EError, "Invalid interaction mode: %s", lowMode.c_str());
        return BSDF::ENull;
      }
    }

    // Photon tracing options
    int maxDepth;
    int minDepth;
    int rrDepth;
    int granularity;
    bool autoCancelGathering;

    // SPPM parameters
    Float alpha;
    Float initialScale;
    Float initialRadius;            // Global radius
    Float epsilonRadius;            // Minimum radius to start with
    int photonCount;
    int maxPasses;
    bool strictNormals;

    // The reconstruction parameters
    bool reconstructL1;
    bool reconstructL2;
    Float reconstructAlpha;
    int dumpIteration; // When we want to do the reconstruction?

    Float bounceRoughness;

    bool showAbsGrad;
    bool showOnlyRecons;

    int interactionMode;
    bool useManifold;
    bool useMIS;
    int debugShift;

    int maxManifoldIterations;
    bool forceBlackPixels;
    bool fluxLightDensity;
    bool directTracing;
    bool nearSpecularDirectImage;
    bool nanCheck;

    // For debugging the rendering using a particular eye subpath length.
    int eyePathMinDepth;
    int eyePathMaxDepth;

    // Set to true to make GPM work as SPPM, i.e., only output throughput images. 
    // Other settings such as min depth and interaction mode is the same. 
    // This is a workaround as supporting interaction mode in original SPPM implementation could 
    // require big modification of the existing vanilla code, e.g., Photon data structure. 
    bool pretendSPPM;

    // In the reference mode, the sampler are initialized
    // randomly to be sure to not have the same
    // sequence of gather points generated
    bool referenceMod;

  void load(const Properties &props) {
        initialScale = props.getFloat("initialScale", 1.f);
        /* Initial photon query radius (0 = infer based on scene size and sensor resolution) */
        initialRadius = props.getFloat("initialRadius", 0);
        /* The minimum non-zero radius that can be used to start density estimation */
        epsilonRadius = props.getFloat("epsilonRadius", 1e-3f);
        /* Alpha parameter from the paper (influences the speed, at which the photon radius is reduced) */
        alpha = props.getFloat("alpha", .7);
        /* Number of photons to shoot in each iteration */
        photonCount = props.getInteger("photonCount", 250000);
        /* Granularity of the work units used in parallelizing the
        particle tracing task (default: choose automatically). */
        granularity = props.getInteger("granularity", 0);
        /* Longest visualized path length (<tt>-1</tt>=infinite). When a positive value is
        specified, it must be greater or equal to <tt>2</tt>, which corresponds to single-bounce
        (direct-only) illumination */
        maxDepth = props.getInteger("maxDepth", -1);
        minDepth = props.getInteger("minDepth", 0);
        /* Depth to start using russian roulette */
        rrDepth = props.getInteger("rrDepth", 3);
        /* Indicates if the gathering steps should be canceled if not enough photons are generated. */
        autoCancelGathering = props.getBoolean("autoCancelGathering", true);
        /* Maximum number of passes to render. -1 renders until the process is stopped. */
        maxPasses = props.getInteger("maxPasses", -1);
        
        if (maxDepth <= 1 && maxDepth != -1)
            SLog(EError, "Maximum depth must be set to \"2\" or higher!");
        if (maxPasses <= 0 && maxPasses != -1)
            SLog(EError, "Maximum number of passes must either be set to \"-1\" or \"1\" or higher!");

        strictNormals = props.getBoolean("strictNormals", false);

        // Ddump a hdr file every N frames
        dumpIteration = props.getInteger("dumpIteration", 5);

        // Reconstruction parameters
        reconstructL1 = props.getBoolean("reconstructL1", false);
        reconstructL2 = props.getBoolean("reconstructL2", true);      // L2 by default
        reconstructAlpha = props.getFloat("reconstructAlpha", Float(0.2));
        
        showOnlyRecons = props.getBoolean("showOnlyRecons", true);
        showAbsGrad = props.getBoolean("showAbsGrad", false);
        
        // Vertex classification 
        bounceRoughness = props.getFloat("bounceRoughness", 0.001);
        if (bounceRoughness <= 0.0 || bounceRoughness > 1.0) {
            SLog(EError, "Bad roughtness constant: %f", bounceRoughness);
        }

        if (showOnlyRecons && !reconstructL1 && !reconstructL2) {
            SLog(EError, "No reconstruction is available for display");
        }

        // Option for testing our algorithm
        interactionMode = getInteractionMode(props.getString("interactionMode","all"));
        useManifold = props.getBoolean("useManifold", true);

        // MIS selection
        std::string mis = props.getString("useMIS", "area");
        if (mis == "area")          useMIS = true;
        else                        useMIS = false;

        // Shift debugging: To only look at one shift results
        std::string debugShiftStr = props.getString("debugShift", "0");
        if(debugShiftStr == "ME") debugShift = EManifoldShift;
        else if(debugShiftStr == "HV") debugShift = EHalfVectorShift;
        else if(debugShiftStr == "DIFFUSE") debugShift = EDiffuseShift;
        else debugShift = -1;


        maxManifoldIterations = props.getInteger("maxManifoldIterations", 5);
        forceBlackPixels = props.getBoolean("forceBlackPixels", false);
        fluxLightDensity = props.getBoolean("fluxLightDensity", true);

        /* Skip gradient computation for pure specular path. Final image = direct image + reconstruction image. */
        directTracing = props.getBoolean("directTracing", false);
        /* Consider near-specular path as pure specular and skip computing gradients for these. */
        nearSpecularDirectImage = props.getBoolean("nearSpecularDirectImage", false);
        nanCheck = props.getBoolean("nanCheck", false);

        pretendSPPM = props.getBoolean("pretendSPPM", false);

        eyePathMinDepth = props.getInteger("eyePathMinDepth", 0);
        eyePathMaxDepth = props.getInteger("eyePathMaxDepth", -1);

        referenceMod = props.getBoolean("referenceMod", false);
    }
};

enum EPixel {
    ELeft = 0,
    ERight = 1,
    ETop = 2,
    EBottom = 3
};

/// Represents one individual PPM gather point including relevant statistics
struct GatherPoint {
  // Radiance information
  Spectrum weight;                  // weight of the eye subpath

  Spectrum baseFlux;                // total flux of this gather point (eye path weight multiplied)
  Spectrum shiftedFlux[4];          // MIS weighted, total flux of the shifted path (eye path weight multiplied)
  Spectrum weightedBaseFlux[4];     // MIS weighted, total flux of the base path (eye path weight multiplied)

  // Same things for the emssion
  Spectrum currEmission;
  Spectrum shiftedEmitterFlux[4];
  Spectrum weightedEmitterFlux[4];

  //
  bool pureSpecular;
  Spectrum directEmission;
  Spectrum emission;

  // Tracing history 
  Path *path;

  Float pdf[2];                     // For easy switching, we store pdf and Jacobian in both measure.
                                    // Please don't be confused with pdf[ERadiance] and pdf[EImportance] in path vertex.
  Float jacobian[2];

  // Geometry information
  Intersection its;
  Float radius;
  Float N;
  int depth;
  Point2i pos;

  // BSDF information
  int sampledComponent;         // the component index of a multi-lobe BSDF
  float pdfComponent;
  Float scale;

  inline GatherPoint() : weight(0.0f), baseFlux(0.0f), emission(0.0f),
                         N(0.0f),
                         sampledComponent(-1), pdfComponent(1.0f), scale(1.0f),
                         path(NULL), depth(-1), currEmission(0.f), directEmission(0.f), pureSpecular(true)
  { 
      for (int i = 0; i < 4; ++i) {
          shiftedFlux[i] = Spectrum(0.0f);
          weightedBaseFlux[i] = Spectrum(0.0f);
          shiftedEmitterFlux[i] = Spectrum(0.f);
          weightedEmitterFlux[i] = Spectrum(0.f);
      }
      for (int i = 0; i < 2; ++i) {
          pdf[i] = 0.0f;
          jacobian[i] = 1.0f;
      }
  }

  void clear() {

      depth = -1;
      its = Intersection();

  }

  void resetTemp() {

  }

  inline void rescaleRadii(Float v) {
    baseFlux *= v;
    
    for (int i = 0; i < 4; ++i) {
        shiftedFlux[i] *= v;
        weightedBaseFlux[i] *= v;
    }
  }
};

/// Classification of vertices into diffuse and glossy.
enum EVertexType {
  VERTEX_TYPE_GLOSSY,     ///< "Specular" vertex that requires the half-vector duplication shift.
  VERTEX_TYPE_DIFFUSE,     ///< "Non-specular" vertex that is rough enough for the reconnection shift.
  VERTEX_TYPE_INVALID
};

struct VertexClassifier {
  static Float roughnessThreshold;
  /**
   * \brief Classify a pathvertex into diffuse or specular for gradient tracing use.
   *
   * \param bsdfComponentType         The component type (ESmooth or EDelta) being considered.
   *                                  If a BSDF has a smooth and a delta component,
   *                                  the delta will be ignored in the classification
   *                                  if the smooth component is being considered.
   */

  static EVertexType type(const BSDF* bsdf, const Intersection& its, int comp) {
      comp = comp == -1 ? 0 : comp;
      if(comp >= bsdf->getComponentCount()) {
          return VERTEX_TYPE_INVALID; // This can be a quite bit drastic
      } else {
          return getVertexTypeByRoughness(bsdf->getRoughness(its, comp));
      }
  }

  static EVertexType type(const PathVertex &vertex, int comp) {
      if (vertex.getType() == PathVertex::EEmitterSample)
          return VERTEX_TYPE_DIFFUSE;
      return type(vertex.getIntersection().getBSDF(), vertex.getIntersection(), comp );
  }

  static bool enoughRough(const Intersection& its, const BSDF* bsdf) {
      if(bsdf->getComponentCount() == 0) {
          // No BSDF, just said no
          return false;
      }

      for(int i = 0; i < bsdf->getComponentCount(); i++) {
          if(getVertexTypeByRoughness(bsdf->getRoughness(its, i)) == VERTEX_TYPE_DIFFUSE)
              return true;
      }
      // All the component are "glossy", return false
      return false;
  }

  static bool enoughRough(const PathVertex& vertex) {
      if (vertex.getType() != PathVertex::ESurfaceInteraction) {
          SLog(EError, "On the light source?");
          return false;
      }
      const Intersection& its = vertex.getIntersection();
      const BSDF* bsdf = its.getBSDF();
      return enoughRough(its, bsdf);
  }

  /// Returns the vertex type of a vertex by its roughness value.
 private:
  static EVertexType getVertexTypeByRoughness(Float roughness) {
      if (roughness <= roughnessThreshold) {
          return VERTEX_TYPE_GLOSSY;
      }
      else {
          return VERTEX_TYPE_DIFFUSE;
      }
  }
};

struct GPMThreadData {
  ref<ManifoldPerturbation> offsetGenerator;
  MemoryPool pool;

  GPMThreadData(Scene* scene, const GPMConfig& config) {
    ref<Sampler> sampler = static_cast<Sampler *> (PluginManager::getInstance()->
        createObject(MTS_CLASS(Sampler), Properties("independent")));

    offsetGenerator = new ManifoldPerturbation(scene,
                                               sampler,
                                               pool,
                                               0.f, true, true, 0, 0,
                                               config.bounceRoughness, 
                                               config.maxManifoldIterations);
  }
};

MTS_NAMESPACE_END
