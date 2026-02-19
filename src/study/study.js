import { FIXED_PARAMS_KEY, STUDY_SNAPSHOT_MAGIC, STUDY_SNAPSHOT_VERSION } from '../core/constants.js'
import { TrialState } from '../core/enums.js'
import { isPlainObject } from '../core/objectUtils.js'
import { cloneJsonValue } from '../core/snapshotJson.js'
import {
  deserializeSamplerFromSnapshot,
  deserializeTrialFromSnapshot,
  serializeSamplerForSnapshot,
  serializeTrialForSnapshot
} from './snapshotCodec.js'
import { TrialRuntime } from './trialRuntime.js'

export function createFrozenTrial({
  number,
  state,
  params = {},
  distributions = {},
  systemAttrs = {},
  intermediateValues = {},
  value = null,
  values = null
}) {
  return {
    number,
    state,
    params,
    distributions,
    system_attrs: systemAttrs,
    intermediate_values: intermediateValues,
    value,
    values
  }
}

export class Study {
  constructor({ sampler, directions }) {
    this.sampler = sampler
    this.directions = directions
    this.direction = directions[0]
    this.trials = []
  }

  isMultiObjective() {
    return this.directions.length > 1
  }

  enqueueTrial(params) {
    if (!isPlainObject(params)) {
      throw new Error('enqueueTrial expects params to be an object.')
    }

    const frozen = createFrozenTrial({
      number: this.trials.length,
      state: TrialState.WAITING,
      systemAttrs: {
        [FIXED_PARAMS_KEY]: cloneJsonValue(params)
      }
    })
    this.trials.push(frozen)
  }

  ask() {
    let frozen = null
    for (let i = 0; i < this.trials.length; i += 1) {
      if (this.trials[i].state === TrialState.WAITING) {
        frozen = this.trials[i]
        break
      }
    }

    if (frozen === null) {
      frozen = createFrozenTrial({
        number: this.trials.length,
        state: TrialState.RUNNING
      })
      this.trials.push(frozen)
    } else {
      frozen.state = TrialState.RUNNING
      frozen.params = frozen.params || {}
      frozen.distributions = frozen.distributions || {}
      frozen.system_attrs = frozen.system_attrs || {}
      frozen.intermediate_values = frozen.intermediate_values || {}
      frozen.value = frozen.value ?? null
      frozen.values = frozen.values ?? null
    }

    this.sampler.beforeTrial(this, frozen)
    return new TrialRuntime(this, frozen)
  }

  tell(trialRuntime, { value = null, values = null, state = null } = {}) {
    const frozen = trialRuntime instanceof TrialRuntime ? trialRuntime.frozen : trialRuntime
    if (state === null || state === undefined) {
      state = TrialState.COMPLETE
    }

    if (state === TrialState.COMPLETE) {
      if (this.isMultiObjective()) {
        if (!Array.isArray(values)) {
          throw new Error('Multi-objective study requires values array on complete trial.')
        }
        frozen.values = values.slice()
        frozen.value = null
      } else {
        if (typeof value !== 'number') {
          throw new Error('Single-objective study requires numeric value on complete trial.')
        }
        frozen.value = value
        frozen.values = null
      }
    }

    frozen.state = state
    this.sampler.afterTrial(this, frozen, state, frozen.values)
  }

  getTrials({ states = null, useCache = true } = {}) {
    void useCache
    if (states === null) {
      return this.trials
    }
    return this.trials.filter((trial) => states.includes(trial.state))
  }

  serialize() {
    return {
      magic: STUDY_SNAPSHOT_MAGIC,
      version: STUDY_SNAPSHOT_VERSION,
      directions: this.directions.slice(),
      sampler: serializeSamplerForSnapshot(this.sampler),
      trials: this.trials.map((trial) => serializeTrialForSnapshot(trial))
    }
  }

  toJSON() {
    return this.serialize()
  }

  static deserialize(snapshot, options = {}) {
    if (!isPlainObject(snapshot)) {
      throw new Error('Invalid study snapshot: expected an object.')
    }
    if (snapshot.magic !== STUDY_SNAPSHOT_MAGIC) {
      throw new Error(
        `Invalid study snapshot magic "${snapshot.magic}". Expected "${STUDY_SNAPSHOT_MAGIC}".`
      )
    }
    if (snapshot.version !== STUDY_SNAPSHOT_VERSION) {
      throw new Error(
        `Unsupported study snapshot version ${snapshot.version}. Expected ${STUDY_SNAPSHOT_VERSION}.`
      )
    }
    if (!Array.isArray(snapshot.directions) || snapshot.directions.length === 0) {
      throw new Error('Invalid study snapshot: missing directions.')
    }
    if (!Array.isArray(snapshot.trials)) {
      throw new Error('Invalid study snapshot: missing trials array.')
    }

    const sampler = deserializeSamplerFromSnapshot(snapshot.sampler, options)
    const study = new Study({
      sampler,
      directions: snapshot.directions.slice()
    })
    study.trials = snapshot.trials.map((trial) => deserializeTrialFromSnapshot(trial))
    return study
  }

  static fromJSON(snapshot, options = {}) {
    return Study.deserialize(snapshot, options)
  }

  static parse(serializedJson, options = {}) {
    return Study.deserialize(JSON.parse(serializedJson), options)
  }
}

export function serializeStudy(study) {
  if (!(study instanceof Study)) {
    throw new Error('serializeStudy expects an instance of Study.')
  }
  return study.serialize()
}

export function deserializeStudy(snapshot, options = {}) {
  return Study.deserialize(snapshot, options)
}

export function sanitizeParams(params) {
  const out = {}
  for (const [key, value] of Object.entries(params)) {
    if (typeof value === 'number' || typeof value === 'string' || typeof value === 'boolean') {
      out[key] = value
    } else {
      out[key] = String(value)
    }
  }
  return out
}
