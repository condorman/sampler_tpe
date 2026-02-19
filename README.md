### Environment Setup

```bash
python3.13 -m venv env_3.13
source env_3.13/bin/activate
pip install -U pip
pip install -r requirements.in
```

### Study Persistence (Serialize / Deserialize)

`Study` can be serialized to a plain JSON-compatible snapshot and restored later.

```js
import { Study, createTPESampler } from './src/tpe/optuna_tpe.js'

const sampler = createTPESampler({ seed: 42 })
const study = new Study({ sampler, directions: ['minimize'] })

const trial = study.ask()
trial.suggestFloat('x', -5, 5)
study.tell(trial, { value: 1.23 })

// Persist snapshot.
const snapshot = study.serialize()
const json = JSON.stringify(snapshot)

// Restore snapshot.
const restored = Study.parse(json)
```

For custom sampler functions (`gamma`, `weights`, `constraintsFunc`, `categoricalDistanceFunc`),
you must provide them while restoring:

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

### Golden Output Tests

The golden tests compare trial-by-trial output against the fixture in
`fixtures/golden-tpe/tpe_golden.json`.

#### 1) Regenerate fixture (Python / Optuna reference)

```bash
./env_3.13/bin/python generate_tpe_golden.py
```

#### 2) Run golden test (Vitest)

```bash
npx vitest run tpeCore.golden.test.js
```

`tpeCore.golden.runner.js` is the fixed bridge used by the golden test, and it
uses the exported API from `src/tpe/optuna_tpe.js`.

#### 3) Quick smoke flow

```bash
./env_3.13/bin/python generate_tpe_golden.py
npx vitest run tpeCore.golden.test.js
```
