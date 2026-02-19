# optuna-tpe-js

JavaScript implementation of Optuna-like TPE sampler and `Study` primitives.

## Install

```bash
npm install optuna-tpe-js
```

## Usage

```js
import { Study, createTPESampler, TrialState, StudyDirection } from 'optuna-tpe-js'

const sampler = createTPESampler({ seed: 42 })
const study = new Study({ sampler, directions: [StudyDirection.MINIMIZE] })

const trial = study.ask()
const x = trial.suggestFloat('x', -5, 5)
study.tell(trial, { value: x * x })
```

## Study Persistence (Serialize / Deserialize)

`Study` can be serialized to a plain JSON-compatible snapshot and restored later.

```js
import { Study, createTPESampler } from 'optuna-tpe-js'

const sampler = createTPESampler({ seed: 42 })
const study = new Study({ sampler, directions: ['minimize'] })

const trial = study.ask()
trial.suggestFloat('x', -5, 5)
study.tell(trial, { value: 1.23 })

const snapshot = study.serialize()
const json = JSON.stringify(snapshot)
const restored = Study.parse(json)
```

For custom sampler functions (`gamma`, `weights`, `constraintsFunc`, `categoricalDistanceFunc`),
provide them during restore:

```js
const restored = Study.deserialize(snapshot, {
  samplerFunctions: {
    gamma: myGammaFn,
    weights: myWeightsFn,
    constraintsFunc: myConstraintsFn,
    categoricalDistanceFunc: { myParam: myDistanceFn }
  }
})
```

## Development Setup

```bash
npm install
python3.13 -m venv env_3.13
source env_3.13/bin/activate
pip install -U pip
pip install -r requirements.in
```

## Golden Output Tests

The golden tests compare trial-by-trial output against `fixtures/golden-tpe/tpe_golden.json`.
`tpeCore.golden.runner.js` is the fixed bridge and uses `src/optuna_tpe.js`.

```bash
npm run golden:generate
npm run test:golden
```

## Publish to NPM

1. Update `name` and `version` in `package.json`.
2. Validate published contents with `npm run pack:check`.
3. Run tests with `npm test`.
4. Login with `npm login`.
5. Publish with `npm publish` (or `npm publish --access public` for scoped packages).
