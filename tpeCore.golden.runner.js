import {
  Study,
  TrialState,
  createTPESampler,
  sanitizeParams
} from './src/optuna_tpe.js'

function objectiveSingle(params) {
  const catMap = { a: 0, b: 1, c: 2 }
  const x = Number(params.x)
  const y = Number(params.y)
  const mode = params.mode
  const logU = Number(params.log_u)
  const logI = Number(params.log_i)
  const catScore = catMap[mode] ?? 0
  return (
    (x - 0.3) ** 2 +
    (y - 5.0) ** 2 +
    Math.log(logU) ** 2 +
    Math.log(logI) ** 2 +
    catScore * 0.1
  )
}

function objectiveSingleNumeric(params) {
  const x = Number(params.x)
  const y = Number(params.y)
  const logU = Number(params.log_u)
  return (x - 0.3) ** 2 + (y - 5.0) ** 2 + Math.log(logU) ** 2
}

function objectiveMulti(params) {
  const group = params.group ?? 'a'
  const common = Number(params.common ?? 0)
  let x
  let y
  if (group === 'a') {
    x = Number(params.a1 ?? 0)
    y = common
  } else {
    x = Number(params.b1 ?? 0)
    y = common
  }
  return [x * x + (y - 2.0) ** 2, (x - 1.5) ** 2 + (y + 0.25) ** 2]
}

function constraintFunc(params) {
  const x = Number(params.x)
  const y = Number(params.y)
  return [Math.max(0, x - 1.5), Math.max(0, 0.5 - y)]
}

function suggestCore(trial) {
  const params = {}
  params.x = trial.suggestFloat('x', -5.0, 5.0)
  params.y = trial.suggestInt('y', 1, 9, { step: 2 })
  params.mode = trial.suggestCategorical('mode', ['a', 'b', 'c'])
  params.log_u = trial.suggestFloat('log_u', 1e-3, 1e2, { log: true })
  params.log_i = trial.suggestInt('log_i', 1, 64, { log: true })
  return params
}

function enqueuedCoreParams() {
  return [
    {
      x: -4.25,
      y: 1,
      mode: 'c',
      log_u: 0.0015,
      log_i: 2
    },
    {
      x: 4.9,
      y: 9,
      mode: 'a',
      log_u: 12.5,
      log_i: 16
    },
    {
      x: 0.0,
      y: 5,
      mode: 'b',
      log_u: 0.1,
      log_i: 4
    }
  ]
}

function suggestNumeric(trial) {
  const params = {}
  params.x = trial.suggestFloat('x', -5.0, 5.0)
  params.y = trial.suggestFloat('y', -2.0, 8.0)
  params.log_u = trial.suggestFloat('log_u', 1e-3, 1e2, { log: true })
  return params
}

function suggestGroup(trial) {
  const params = {}
  params.group = trial.suggestCategorical('group', ['a', 'b'])
  params.common = trial.suggestFloat('common', -1.0, 1.0)
  if (params.group === 'a') {
    params.a1 = trial.suggestFloat('a1', -2.0, 2.0)
  } else {
    params.b1 = trial.suggestFloat('b1', 0.0, 4.0)
  }
  return params
}

function objectiveSingleGroup(params) {
  const values = objectiveMulti(params)
  return values[0] + 0.5 * values[1]
}

function gammaCustom(nTrials) {
  return Math.min(32, Math.ceil(0.2 * nTrials))
}

function weightsCustom(nObservations) {
  const out = []
  for (let i = 0; i < nObservations; i += 1) {
    out.push((i + 1) * (i + 1))
  }
  return out
}

function runSingleObjectiveNumericWithSampler(seed, nTrials, directions, samplerOverrides = {}) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    ...samplerOverrides
  })
  const study = new Study({ sampler, directions })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestNumeric(trial)
    const value = objectiveSingleNumeric(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function evaluateWithPruning(trial, params) {
  const base = objectiveSingle(params)
  const intermediates = []
  for (const step of [1, 2, 3]) {
    const intermediate = base + step * 0.1
    trial.report(intermediate, step)
    intermediates.push([step, intermediate])
    if (step === 2 && intermediate > 1.25) {
      return {
        state: 'pruned',
        value: null,
        intermediateValues: intermediates
      }
    }
  }

  return {
    state: 'complete',
    value: base,
    intermediateValues: intermediates
  }
}

function makeRecord({
  number,
  params,
  state,
  value = null,
  values = null,
  intermediateValues = [],
  constraint = null
}) {
  return {
    number,
    params: sanitizeParams(params),
    state,
    value,
    values,
    intermediate_values: intermediateValues,
    constraint
  }
}

function runSingleObjective(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestCore(trial)
    const value = objectiveSingle(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runSingleObjectiveEnqueuedTrials(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (const params of enqueuedCoreParams()) {
    study.enqueueTrial(params)
  }

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestCore(trial)
    const value = objectiveSingle(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runSingleObjectiveFailures(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 30,
    nEiCandidates: 24
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestCore(trial)
    const value = objectiveSingle(params)

    if (trial.number % 6 === 1) {
      study.tell(trial, { state: TrialState.FAIL })
      records.push(
        makeRecord({
          number: trial.number,
          params,
          state: 'fail',
          value: null,
          values: null,
          intermediateValues: [],
          constraint: null
        })
      )
    } else {
      study.tell(trial, { value })
      records.push(
        makeRecord({
          number: trial.number,
          params,
          state: 'complete',
          value,
          values: null,
          intermediateValues: [],
          constraint: null
        })
      )
    }
  }

  return records
}

function runSingleObjectiveMaximizeNumeric(seed, nTrials) {
  return runSingleObjectiveNumericWithSampler(seed, nTrials, ['maximize'])
}

function runSingleObjectivePriorWeight(seed, nTrials) {
  return runSingleObjectiveNumericWithSampler(seed, nTrials, ['minimize'], {
    priorWeight: 0.2
  })
}

function runSingleObjectiveMagicClipEndpoints(seed, nTrials) {
  return runSingleObjectiveNumericWithSampler(seed, nTrials, ['minimize'], {
    considerMagicClip: false,
    considerEndpoints: true
  })
}

function runSingleObjectiveGammaCustom(seed, nTrials) {
  return runSingleObjectiveNumericWithSampler(seed, nTrials, ['minimize'], {
    gamma: gammaCustom
  })
}

function runSingleObjectiveWeightsCustom(seed, nTrials) {
  return runSingleObjectiveNumericWithSampler(seed, nTrials, ['minimize'], {
    weights: weightsCustom
  })
}

function runSingleObjectiveEiCandidatesCustom(seed, nTrials) {
  return runSingleObjectiveNumericWithSampler(seed, nTrials, ['minimize'], {
    nEiCandidates: 64
  })
}

function runSingleObjectiveHighStartup(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 30,
    nEiCandidates: 24
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestCore(trial)
    const value = objectiveSingle(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runSingleObjectiveMultivariate(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    multivariate: true
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestCore(trial)
    const value = objectiveSingle(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runSingleObjectiveGroup(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    multivariate: true,
    group: true
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestGroup(trial)
    const value = objectiveSingleGroup(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runSingleObjectiveDynamicIndependent(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 30,
    nEiCandidates: 24
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestGroup(trial)
    const value = objectiveSingleGroup(params)
    study.tell(trial, { value })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runMultiObjective(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    multivariate: true,
    group: true
  })
  const study = new Study({ sampler, directions: ['minimize', 'minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestGroup(trial)
    const values = objectiveMulti(params)
    study.tell(trial, { values })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value: null,
        values: [values[0], values[1]],
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runMultiObjectiveDynamicIndependent(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 30,
    nEiCandidates: 24
  })
  const study = new Study({ sampler, directions: ['minimize', 'minimize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestGroup(trial)
    const values = objectiveMulti(params)
    study.tell(trial, { values })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value: null,
        values: [values[0], values[1]],
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runMultiObjectiveMixedDirections(seed, nTrials) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    multivariate: true,
    group: true
  })
  const study = new Study({ sampler, directions: ['minimize', 'maximize'] })
  const records = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestGroup(trial)
    const values = objectiveMulti(params)
    study.tell(trial, { values })

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value: null,
        values: [values[0], values[1]],
        intermediateValues: [],
        constraint: null
      })
    )
  }

  return records
}

function runConstantLiarDelayedSingle(seed, nTrials, tellLag) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    constantLiar: true
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []
  const pending = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestNumeric(trial)
    const value = objectiveSingleNumeric(params)

    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: 'complete',
        value,
        values: null,
        intermediateValues: [],
        constraint: null
      })
    )

    pending.push({ trial, value })
    if (pending.length > tellLag) {
      const oldest = pending.shift()
      study.tell(oldest.trial, { value: oldest.value })
    }
  }

  while (pending.length > 0) {
    const oldest = pending.shift()
    study.tell(oldest.trial, { value: oldest.value })
  }

  return records
}

function runConstraintsPruningConstantLiar(seed, nTrials, tellLag) {
  const sampler = createTPESampler({
    seed,
    nStartupTrials: 10,
    nEiCandidates: 24,
    constantLiar: true,
    constraintsFunc: (frozenTrial) => constraintFunc(frozenTrial.params)
  })
  const study = new Study({ sampler, directions: ['minimize'] })
  const records = []
  const pending = []

  for (let i = 0; i < nTrials; i += 1) {
    const trial = study.ask()
    const params = suggestCore(trial)
    const result = evaluateWithPruning(trial, params)

    const constraint = result.state === 'complete' ? constraintFunc(params) : null
    records.push(
      makeRecord({
        number: trial.number,
        params,
        state: result.state,
        value: result.value,
        values: null,
        intermediateValues: result.intermediateValues,
        constraint
      })
    )

    pending.push({ trial, state: result.state, value: result.value })

    if (pending.length > tellLag) {
      const oldest = pending.shift()
      if (oldest.state === 'pruned') {
        study.tell(oldest.trial, { state: TrialState.PRUNED })
      } else {
        study.tell(oldest.trial, { value: oldest.value })
      }
    }
  }

  while (pending.length > 0) {
    const oldest = pending.shift()
    if (oldest.state === 'pruned') {
      study.tell(oldest.trial, { state: TrialState.PRUNED })
    } else {
      study.tell(oldest.trial, { value: oldest.value })
    }
  }

  return records
}

export async function runGoldenScenario({ name, seed, nTrials, tellLag }) {
  if (name === 'core_single_objective') {
    return runSingleObjective(seed, nTrials)
  }
  if (name === 'single_objective_enqueued_trials') {
    return runSingleObjectiveEnqueuedTrials(seed, nTrials)
  }
  if (name === 'single_objective_failures') {
    return runSingleObjectiveFailures(seed, nTrials)
  }
  if (name === 'single_objective_maximize_numeric') {
    return runSingleObjectiveMaximizeNumeric(seed, nTrials)
  }
  if (name === 'single_objective_prior_weight') {
    return runSingleObjectivePriorWeight(seed, nTrials)
  }
  if (name === 'single_objective_magic_clip_endpoints') {
    return runSingleObjectiveMagicClipEndpoints(seed, nTrials)
  }
  if (name === 'single_objective_gamma_custom') {
    return runSingleObjectiveGammaCustom(seed, nTrials)
  }
  if (name === 'single_objective_weights_custom') {
    return runSingleObjectiveWeightsCustom(seed, nTrials)
  }
  if (name === 'single_objective_n_ei_candidates_custom') {
    return runSingleObjectiveEiCandidatesCustom(seed, nTrials)
  }
  if (name === 'single_objective_high_startup') {
    return runSingleObjectiveHighStartup(seed, nTrials)
  }
  if (name === 'single_objective_multivariate') {
    return runSingleObjectiveMultivariate(seed, nTrials)
  }
  if (name === 'single_objective_group') {
    return runSingleObjectiveGroup(seed, nTrials)
  }
  if (name === 'single_objective_dynamic_independent') {
    return runSingleObjectiveDynamicIndependent(seed, nTrials)
  }
  if (name === 'multi_objective_group') {
    return runMultiObjective(seed, nTrials)
  }
  if (name === 'multi_objective_dynamic_independent') {
    return runMultiObjectiveDynamicIndependent(seed, nTrials)
  }
  if (name === 'multi_objective_mixed_directions') {
    return runMultiObjectiveMixedDirections(seed, nTrials)
  }
  if (name === 'constant_liar_delayed_single') {
    return runConstantLiarDelayedSingle(seed, nTrials, tellLag)
  }
  if (name === 'constraints_pruning_constant_liar') {
    return runConstraintsPruningConstantLiar(seed, nTrials, tellLag)
  }

  throw new Error(`Unknown scenario: ${name}`)
}
