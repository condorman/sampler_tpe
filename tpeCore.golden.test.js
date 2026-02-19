import { describe, it, expect } from 'vitest'
import fs from 'fs'
import path from 'path'
import { fileURLToPath, pathToFileURL } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const FIXTURE_PATH = path.join(__dirname, 'fixtures', 'golden-tpe', 'tpe_golden.json')
const DEFAULT_ADAPTER_PATH = path.join(__dirname, 'tpeCore.golden.adapter.js')
const ADAPTER_PATH = process.env.TPE_GOLDEN_ADAPTER
  ? path.resolve(process.cwd(), process.env.TPE_GOLDEN_ADAPTER)
  : DEFAULT_ADAPTER_PATH
const ABS_EPSILON = 1e-12
const REL_EPSILON = 1e-9

const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf8'))

let runGoldenScenario = null
let adapterLoadError = null

try {
  const adapterModule = await import(pathToFileURL(ADAPTER_PATH).href)
  if (typeof adapterModule.runGoldenScenario !== 'function') {
    throw new Error(
      `Missing export "runGoldenScenario" in ${ADAPTER_PATH}. ` +
        'Expected signature: async ({ name, seed, nTrials, tellLag, objectiveDirections }) => TrialRecord[]'
    )
  }
  runGoldenScenario = adapterModule.runGoldenScenario
} catch (error) {
  adapterLoadError = error
}

function normalizeTrial(trial) {
  return {
    number: trial.number,
    params: trial.params ?? {},
    state: trial.state,
    value: trial.value ?? null,
    values: trial.values ?? null,
    intermediate_values: trial.intermediate_values ?? [],
    constraint: trial.constraint ?? null
  }
}

function expectNearlyEqual(actual, expected, context) {
  if (expected === null) {
    expect(actual, context).toBeNull()
    return
  }

  if (typeof expected === 'number') {
    expect(typeof actual, context).toBe('number')
    const diff = Math.abs(actual - expected)
    const tolerance = Math.max(ABS_EPSILON, Math.abs(expected) * REL_EPSILON)
    expect(
      diff <= tolerance,
      `${context}: expected=${expected}, actual=${actual}, diff=${diff}, tolerance=${tolerance}`
    ).toBe(true)
    return
  }

  if (Array.isArray(expected)) {
    expect(Array.isArray(actual), `${context}: expected array`).toBe(true)
    expect(actual.length, `${context}: array length`).toBe(expected.length)
    for (let i = 0; i < expected.length; i += 1) {
      expectNearlyEqual(actual[i], expected[i], `${context}[${i}]`)
    }
    return
  }

  if (typeof expected === 'object') {
    expect(typeof actual, `${context}: expected object`).toBe('object')
    const expectedKeys = Object.keys(expected).sort()
    const actualKeys = Object.keys(actual).sort()
    expect(actualKeys, `${context}: object keys`).toEqual(expectedKeys)
    for (const key of expectedKeys) {
      expectNearlyEqual(actual[key], expected[key], `${context}.${key}`)
    }
    return
  }

  expect(actual, context).toBe(expected)
}

function expectSameTrial(actualTrial, expectedTrial, context) {
  const actual = normalizeTrial(actualTrial)
  const expected = normalizeTrial(expectedTrial)
  expectNearlyEqual(actual, expected, context)
}

describe('tpe golden fixture', () => {
  it('contains scenarios', () => {
    expect(Array.isArray(fixture.scenarios)).toBe(true)
    expect(fixture.scenarios.length).toBeGreaterThan(0)
  })
})

describe('tpe golden adapter', () => {
  it('is loadable', () => {
    if (adapterLoadError) {
      throw new Error(
        `Unable to load adapter from ${ADAPTER_PATH}: ${adapterLoadError.message}`
      )
    }
    expect(typeof runGoldenScenario).toBe('function')
  })
})

const parityDescribe = adapterLoadError ? describe.skip : describe

parityDescribe('TPE parity against Optuna golden fixture', () => {
  for (const scenario of fixture.scenarios) {
    describe(`scenario: ${scenario.name}`, () => {
      for (const run of scenario.runs) {
        it(`matches seed=${run.seed}`, async () => {
          const actualTrials = await runGoldenScenario({
            name: scenario.name,
            seed: run.seed,
            nTrials: run.trials.length,
            tellLag: scenario.tellLag,
            objectiveDirections: scenario.objectiveDirections
          })

          expect(Array.isArray(actualTrials)).toBe(true)
          expect(actualTrials.length).toBe(run.trials.length)

          for (let idx = 0; idx < run.trials.length; idx += 1) {
            expectSameTrial(
              actualTrials[idx],
              run.trials[idx],
              `${scenario.name}/seed=${run.seed}/trial=${idx}`
            )
          }
        })
      }
    })
  }
})
