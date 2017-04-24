#pragma once

#include <mitsuba/render/particleproc.h>
#include <mitsuba/bidir/mempool.h>

#include "gpm_photonmap.h"

MTS_NAMESPACE_BEGIN

class GradientPhotonProcess : public ParticleProcess {
public:
  /**
   * Create a new process for parallel photon gathering
   * \param type
   *     Specifies the type of requested photons (surface/caustic/volume)
   * \param photonCount
   *     Specifies the number of requested photons
   * \param granularity
   *     Size of the internally used work units (in photons)
   * \param isLocal
   *     Should the parallel process only be executed locally? (sending
   *     photons over the network may be unnecessary and wasteful)
   * \param autoCancel
   *     Indicates if the gathering process should be canceled if there
   *     are not enough photons generated
   * \param progressReporterPayload
   *    Custom pointer payload to be delivered with progress messages
   */
  GradientPhotonProcess(size_t photonCount,
      size_t granularity, int maxDepth, int rrDepth, bool isLocal,
      bool autoCancel, const void *progressReporterPayload,
      std::vector<MemoryPool>& memoryPools, int minDepth);

  /**
   * Once the process has finished, this returns a reference
   * to the (still unbalanced) photon map
   */
  inline GPhotonMap *getPhotonMap() { return m_photonMap; }

  /**
   * \brief Return the number of discarded photons
   *
   * Due to asynchronous processing, some excess photons
   * will generally be produced. This function returns the number
   * of excess photons that had to be discarded. If this is too
   * high, the granularity should be decreased.
   */
  inline size_t getExcessPhotons() const { return m_excess; }

  /**
   * \brief Lists the nuber of particles that had to be shot
   * in order to fill the photon map.
   */
  inline size_t getShotParticles() const { return m_numShot; }

  // ======================================================================
  /// @{ \name ParallelProcess implementation
  // ======================================================================

  bool isLocal() const;
  ref<WorkProcessor> createWorkProcessor() const;
  void processResult(const WorkResult *wr, bool cancelled);
  EStatus generateWork(WorkUnit *unit, int worker);

  /// @}
  // ======================================================================

  MTS_DECLARE_CLASS()
protected:
  /// Virtual destructor
  virtual ~GradientPhotonProcess() { }

  /**
   * \brief Checks if the configuration of needed, generated and shot
   * photons indicates an unsuccessful progress of the gathering. This
   * check is taken from PBRT.
   */
  inline bool unsuccessful(size_t needed, size_t gen, size_t shot) {
    return (gen < needed && (gen == 0 || gen < shot/1024));
  }
protected:
  ref<GPhotonMap> m_photonMap;
  size_t m_photonCount;
  int m_maxDepth;
  int m_rrDepth;
  bool m_isLocal;
  bool m_autoCancel;
  size_t m_excess, m_numShot;

  // Pools used to generate light paths
  std::vector<MemoryPool>& m_pools;
  mutable int m_workerID;
  int m_minDepth;
};


MTS_NAMESPACE_END