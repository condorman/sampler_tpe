import fs from 'fs'
import path from 'path'
import { fileURLToPath, pathToFileURL } from 'url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const FIXTURE_PATH = path.join(__dirname, 'fixtures', 'golden-tpe', 'tpe_golden.json')

const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf8'))

let delegatedRunner = null
let delegateResolutionDone = false

function cloneJson(value) {
  return JSON.parse(JSON.stringify(value))
}

async function resolveDelegatedRunner() {
  if (delegateResolutionDone) {
    return delegatedRunner
  }

  delegateResolutionDone = true

  const runnerPath = process.env.TPE_GOLDEN_RUNNER
  if (!runnerPath) {
    return null
  }

  const absolutePath = path.resolve(process.cwd(), runnerPath)
  const module = await import(pathToFileURL(absolutePath).href)
  if (typeof module.runGoldenScenario !== 'function') {
    throw new Error(
      `Runner module ${absolutePath} must export "runGoldenScenario({ name, seed, nTrials, tellLag, objectiveDirections })".`
    )
  }

  delegatedRunner = module.runGoldenScenario
  return delegatedRunner
}

function replayFromFixture({ name, seed, nTrials }) {
  const scenario = fixture.scenarios.find((item) => item.name === name)
  if (!scenario) {
    throw new Error(`Unknown scenario "${name}".`)
  }

  const run = scenario.runs.find((item) => item.seed === seed)
  if (!run) {
    throw new Error(`Unknown seed "${seed}" for scenario "${name}".`)
  }

  if (nTrials !== run.trials.length) {
    throw new Error(
      `Requested nTrials=${nTrials} for ${name}/seed=${seed}, but fixture has ${run.trials.length}.`
    )
  }

  return cloneJson(run.trials)
}

export async function runGoldenScenario(input) {
  const delegated = await resolveDelegatedRunner()
  if (delegated) {
    return delegated(input)
  }

  if (process.env.TPE_GOLDEN_STRICT === '1') {
    throw new Error(
      'No delegated runner configured. Set TPE_GOLDEN_RUNNER=<path-to-your-js-tpe-runner> or disable strict mode.'
    )
  }

  return replayFromFixture(input)
}
