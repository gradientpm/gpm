#include <mitsuba/bidir/path.h>
#include <mitsuba/render/range.h>

#include "gpm_proc.h"

MTS_NAMESPACE_BEGIN

/**
 * \brief This work result implementation stores a sequence of photons, which can be
 * sent over the wire as needed.
 *
 * It is used to implement parallel networked photon tracing passes.
 */
class PhotonPathVector : public WorkResult {
 public:
  PhotonPathVector() {
    idWorker = -1;
  }

  inline void addPath(Path* p) {
    m_lightPaths.push_back(p);
  }

  inline size_t getLightPathCount() const {
    return m_lightPaths.size();
  }

  inline void clear() {
    m_lightPaths.clear();
  }

  inline const Path* operator[](size_t index) const {
    return m_lightPaths[index];
  }

  void load(Stream *stream) {
    SLog(EError, "Impossible to stream GPM");
  }

  void save(Stream *stream) const {
    SLog(EError, "Impossible to stream GPM");
  }

  std::string toString() const {
    std::ostringstream oss;
    oss << "PhotonVector[size=" << m_lightPaths.size() << "]";
    return oss.str();
  }

  int idWorker;

  MTS_DECLARE_CLASS()
 protected:
  // Virtual destructor
  virtual ~PhotonPathVector() { }
 private:
  std::vector<Path*> m_lightPaths;
};

/**
 * This class does the actual photon tracing work
 */
class GradientPhotonWorker: public WorkProcessor {
 public:
  GradientPhotonWorker(int maxDepth, int rrDepth,
                      int workerID, std::vector<MemoryPool>& pools,
                      int minDepth):
      m_maxDepth(maxDepth), m_rrDepth(rrDepth), m_workerID(workerID),
      m_workerPool(&pools[workerID]), m_minDepth(minDepth)
  { }

  GradientPhotonWorker(Stream *stream, InstanceManager *manager)
      : WorkProcessor(stream, manager) {
    SLog(EError, "No serialization for GPM");
  }

  ref<WorkProcessor> clone() const {
    // This is not possible to ensure the uniqueness
    // of the worker ID
    Log(EError, "No suport of worker cloning ... ");
    return 0;
  }

  void serialize(Stream *stream, InstanceManager *manager) const {
    SLog(EError, "No serialization GPM");
  }

  ref<WorkResult> createWorkResult() const {
    return new PhotonPathVector();
  }

  ref<WorkUnit> createWorkUnit() const {
    return new RangeWorkUnit();
  }

  void prepare() {
    Scene *scene = static_cast<Scene *>(getResource("scene"));
    m_scene = new Scene(scene);
    m_sampler = static_cast<Sampler *>(getResource("sampler"));
    Sensor *newSensor = static_cast<Sensor *>(getResource("sensor"));
    m_scene->removeSensor(scene->getSensor());
    m_scene->addSensor(newSensor);
    m_scene->setSensor(newSensor);
    m_scene->initializeBidirectional();
  }

  void process(const WorkUnit *workUnit, WorkResult *workResult,
               const bool &stop) {
    // Get the work task
    const RangeWorkUnit *range = static_cast<const RangeWorkUnit *>(workUnit);

    // Get the work results and clear it from previous
    m_workResult = static_cast<PhotonPathVector *>(workResult);
    m_workResult->clear();
    m_workResult->idWorker = m_workerID; // Attach the worker ID

    m_sampler->generate(Point2i(0));

    for (size_t index = range->getRangeStart(); index <= range->getRangeEnd() && !stop; ++index) {
      // Generate light path
      Path * lPath = new Path;
      lPath->initialize(m_scene, 0.f, EImportance, *m_workerPool);
      int k = lPath->randomWalk(m_scene, m_sampler, m_maxDepth, m_rrDepth, EImportance, *m_workerPool);

      m_workResult->addPath(lPath); 
    }

    m_workResult = NULL;
  }

  MTS_DECLARE_CLASS()
 protected:
  /// Virtual destructor
  virtual ~GradientPhotonWorker() { }
 protected:
  ref<Scene> m_scene;
  ref<Sampler> m_sampler;
  int m_maxDepth;
  int m_rrDepth;

  int m_workerID;
  ref<PhotonPathVector> m_workResult;
  MemoryPool* m_workerPool;
  int m_minDepth;
};

GradientPhotonProcess::GradientPhotonProcess(size_t photonCount,
    size_t granularity, int maxDepth, int rrDepth, bool isLocal, bool autoCancel,
    const void *progressReporterPayload, std::vector<MemoryPool>& pools,
    int minDepth)
: ParticleProcess(ParticleProcess::EGather, photonCount, granularity, "Gathering photons",
  progressReporterPayload), m_photonCount(photonCount), m_maxDepth(maxDepth),
  m_rrDepth(rrDepth),  m_isLocal(isLocal), m_autoCancel(autoCancel), m_excess(0), m_numShot(0),
  m_pools(pools), m_minDepth(minDepth) {
  m_photonMap = new GPhotonMap(photonCount, m_minDepth);
  m_workerID = 0;
}

bool GradientPhotonProcess::isLocal() const {
  return m_isLocal;
}

ref<WorkProcessor> GradientPhotonProcess::createWorkProcessor() const {
  return new GradientPhotonWorker(m_maxDepth, m_rrDepth,
                                  m_workerID++, m_pools,
                                  m_minDepth);
}

void GradientPhotonProcess::processResult(const WorkResult *wr, bool cancelled) {
  if (cancelled)
    return;
  const PhotonPathVector &vec = *static_cast<const PhotonPathVector *>(wr);
  LockGuard lock(m_resultMutex);

  size_t nPhotons = 0;
  size_t nPaths = 0;
  for (size_t i = 0; i < vec.getLightPathCount(); ++i) {
    
    int nPPushed = m_photonMap->tryAppend(vec.idWorker, vec[i]);
    if(nPPushed >= 0) {
       ++nPaths;            // Excess light paths are not considered in normalization
       nPhotons += nPPushed;
    }

    // If it is full just skip
    if (nPPushed == -1 || nPPushed == 0) {
      if(nPPushed == -1) {
        m_excess++;         // Count excessive light paths
      }

      // Free memory for excessive paths and path with no photons added
      Path* currPath = const_cast<Path*>(vec[i]);
      currPath->release(m_pools[vec.idWorker]);
      delete currPath;
    }

  }

  m_numShot += nPaths;
  increaseResultCount(nPhotons);
}

ParallelProcess::EStatus GradientPhotonProcess::generateWork(WorkUnit *unit, int worker) {
  /* Use the same approach as PBRT for auto canceling */
  LockGuard lock(m_resultMutex);
  if (m_autoCancel && m_numShot > 100000
      && unsuccessful(m_photonCount, m_photonMap->size(), m_numShot)) {
    Log(EInfo, "Not enough photons could be collected, giving up");
    return EFailure;
  }

  return ParticleProcess::generateWork(unit, worker);
}

MTS_IMPLEMENT_CLASS(GradientPhotonProcess, false, ParticleProcess)
MTS_IMPLEMENT_CLASS_S(GradientPhotonWorker, false, ParticleTracer)
MTS_IMPLEMENT_CLASS(PhotonPathVector, false, WorkResult)

MTS_NAMESPACE_END