import { clip, roundToNearestEven } from '../core/numberUtils.js'
import {
  logGaussMass,
  truncnormLogpdf,
  truncnormPpf
} from '../math/truncnorm.js'

export class MixtureOfProductDistribution {
  constructor(weights, distributions) {
    this.weights = weights
    this.distributions = distributions
  }

  sample(rng, batchSize) {
    const activeIndices = rng.choiceWeighted(this.weights, batchSize)
    const ret = Array.from({ length: batchSize }, () =>
      new Array(this.distributions.length).fill(0)
    )

    const discNumerical = []
    const numericalDefs = []
    const numericalColumns = []
    const logNumericalColumns = []
    const lowsNumeric = []
    const highsNumeric = []

    for (let col = 0; col < this.distributions.length; col += 1) {
      const distDef = this.distributions[col]
      const d = distDef.distribution
      if (d.kind === 'categorical') {
        const rndQuantile = rng.rand(batchSize)
        for (let row = 0; row < batchSize; row += 1) {
          const weights = d.weights[activeIndices[row]]
          let cum = 0
          let choice = 0
          for (; choice < weights.length; choice += 1) {
            cum += weights[choice]
            if (choice === weights.length - 1) {
              cum = 1
            }
            if (!(cum < rndQuantile[row])) {
              break
            }
          }
          ret[row][col] = choice
        }
        continue
      }

      numericalDefs.push(d)
      numericalColumns.push(col)

      if (d.kind === 'truncnorm') {
        lowsNumeric.push(d.low)
        highsNumeric.push(d.high)
      } else if (d.kind === 'trunclognorm') {
        lowsNumeric.push(Math.log(d.low))
        highsNumeric.push(Math.log(d.high))
        logNumericalColumns.push(col)
      } else if (d.kind === 'discrete_truncnorm') {
        lowsNumeric.push(d.low - d.step / 2)
        highsNumeric.push(d.high + d.step / 2)
        discNumerical.push({
          column: col,
          low: d.low,
          high: d.high,
          step: d.step
        })
      } else if (d.kind === 'discrete_trunclognorm') {
        lowsNumeric.push(Math.log(d.low - d.step / 2))
        highsNumeric.push(Math.log(d.high + d.step / 2))
        logNumericalColumns.push(col)
        discNumerical.push({
          column: col,
          low: d.low,
          high: d.high,
          step: d.step
        })
      } else {
        throw new Error(`Unknown distribution kind: ${d.kind}`)
      }
    }

    if (numericalDefs.length > 0) {
      const activeMus = numericalDefs.map((d) => activeIndices.map((k) => d.mu[k]))
      const activeSigmas = numericalDefs.map((d) => activeIndices.map((k) => d.sigma[k]))

      for (let i = 0; i < numericalDefs.length; i += 1) {
        for (let row = 0; row < batchSize; row += 1) {
          const mu = activeMus[i][row]
          const sigma = activeSigmas[i][row]
          const a = (lowsNumeric[i] - mu) / sigma
          const b = (highsNumeric[i] - mu) / sigma
          const q = rng.randomSample()
          ret[row][numericalColumns[i]] = truncnormPpf(q, a, b) * sigma + mu
        }
      }

      for (const col of logNumericalColumns) {
        for (let row = 0; row < batchSize; row += 1) {
          ret[row][col] = Math.exp(ret[row][col])
        }
      }

      for (const disc of discNumerical) {
        for (let row = 0; row < batchSize; row += 1) {
          const raw = ret[row][disc.column]
          const rounded =
            disc.low + roundToNearestEven((raw - disc.low) / disc.step) * disc.step
          ret[row][disc.column] = clip(rounded, disc.low, disc.high)
        }
      }
    }

    const result = {}
    for (let col = 0; col < this.distributions.length; col += 1) {
      result[this.distributions[col].paramName] = ret.map((row) => row[col])
    }
    return result
  }

  logPdf(samplesByParam) {
    const firstParam = this.distributions[0].paramName
    const nSamples = samplesByParam[firstParam].length
    const out = new Array(nSamples)
    const nWeights = this.weights.length

    for (let s = 0; s < nSamples; s += 1) {
      const weightedLogPdf = new Array(nWeights).fill(0)

      for (const distDef of this.distributions) {
        const d = distDef.distribution
        const x = samplesByParam[distDef.paramName][s]

        if (d.kind === 'categorical') {
          const idx = Math.trunc(x)
          for (let k = 0; k < nWeights; k += 1) {
            weightedLogPdf[k] += Math.log(d.weights[k][idx])
          }
          continue
        }

        if (d.kind === 'truncnorm') {
          for (let k = 0; k < nWeights; k += 1) {
            const a = (d.low - d.mu[k]) / d.sigma[k]
            const b = (d.high - d.mu[k]) / d.sigma[k]
            weightedLogPdf[k] += truncnormLogpdf(x, a, b, d.mu[k], d.sigma[k])
          }
          continue
        }

        if (d.kind === 'trunclognorm') {
          const logX = Math.log(x)
          for (let k = 0; k < nWeights; k += 1) {
            const a = (Math.log(d.low) - d.mu[k]) / d.sigma[k]
            const b = (Math.log(d.high) - d.mu[k]) / d.sigma[k]
            weightedLogPdf[k] += truncnormLogpdf(logX, a, b, d.mu[k], d.sigma[k])
          }
          continue
        }

        if (d.kind === 'discrete_truncnorm') {
          for (let k = 0; k < nWeights; k += 1) {
            const lowLogMass = logGaussMass(
              (d.low - d.step / 2 - d.mu[k]) / d.sigma[k],
              (d.high + d.step / 2 - d.mu[k]) / d.sigma[k]
            )
            const xMass = logGaussMass(
              (x - d.step / 2 - d.mu[k]) / d.sigma[k],
              (x + d.step / 2 - d.mu[k]) / d.sigma[k]
            )
            weightedLogPdf[k] += xMass - lowLogMass
          }
          continue
        }

        if (d.kind === 'discrete_trunclognorm') {
          const logXMinus = Math.log(x - d.step / 2)
          const logXPlus = Math.log(x + d.step / 2)
          const logLowMinus = Math.log(d.low - d.step / 2)
          const logHighPlus = Math.log(d.high + d.step / 2)
          for (let k = 0; k < nWeights; k += 1) {
            const lowLogMass = logGaussMass(
              (logLowMinus - d.mu[k]) / d.sigma[k],
              (logHighPlus - d.mu[k]) / d.sigma[k]
            )
            const xMass = logGaussMass(
              (logXMinus - d.mu[k]) / d.sigma[k],
              (logXPlus - d.mu[k]) / d.sigma[k]
            )
            weightedLogPdf[k] += xMass - lowLogMass
          }
          continue
        }

        throw new Error(`Unknown distribution kind: ${d.kind}`)
      }

      for (let k = 0; k < nWeights; k += 1) {
        weightedLogPdf[k] += Math.log(this.weights[k])
      }

      let maxValue = -Infinity
      for (let k = 0; k < nWeights; k += 1) {
        if (weightedLogPdf[k] > maxValue) {
          maxValue = weightedLogPdf[k]
        }
      }
      if (maxValue === -Infinity) {
        maxValue = 0
      }

      let sumExp = 0
      for (let k = 0; k < nWeights; k += 1) {
        sumExp += Math.exp(weightedLogPdf[k] - maxValue)
      }
      out[s] = Math.log(sumExp) + maxValue
    }

    return out
  }
}
