export const SQRT2 = Math.SQRT2
export const LOG_SQRT_2PI = 0.5 * Math.log(2 * Math.PI)
export const NDTRI_EXP_APPROX_C = Math.sqrt(3) / Math.PI

export function polyEval(coeffs, x) {
  let result = 0
  for (let i = coeffs.length - 1; i >= 0; i -= 1) {
    result = result * x + coeffs[i]
  }
  return result
}

const ERF_COEFF = {
  erx: 8.45062911510467529297e-01,
  efx: 1.28379167095512586316e-01,
  pp: [1.28379167095512558561e-01, -3.2504210724700149937e-01, -2.84817495755985104766e-02, -5.77027029648944159157e-03, -2.37630166566501626084e-05],
  qq: [1, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03, 1.32494738004321644526e-04, -3.9602282787753681232e-06],
  pa: [-2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01, 3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02, -2.166375594868790843e-03],
  qa: [1, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02, 1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.1984499846799107417e-02],
  ra: [-9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e01, -6.23753324503260060396e01, -1.62396669462573470355e02, -1.84605092906711035994e02, -8.12874355063065934246e01, -9.81432934416914548592e00],
  sa: [1, 1.96512716674392571292e01, 1.376577541435190426e02, 4.34565877475229228821e02, 6.45387271733267880336e02, 4.29008140027567833386e02, 1.08635005541779435134e02, 6.57024977031928170135, -6.04244152148580987438e-02],
  rb: [-9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e01, -1.60636384855821916062e02, -6.37566443368389627722e02, -1.02509513161107724954e03, -4.83519191608651397019e02],
  sb: [1, 3.03380607434824582924e01, 3.25792512996573918826e02, 1.53672958608443695994e03, 3.19985821950859553908e03, 2.55305040643316442583e03, 4.74528541206955367215e02, -2.24409524465858183362e01]
}

export function erfScalar(x) {
  if (Number.isNaN(x)) return Number.NaN
  const sign = x < 0 ? -1 : 1
  const a = Math.abs(x)

  if (a >= 6) return sign
  if (a < 2 ** -28) return sign * ((1 + ERF_COEFF.efx) * a)
  if (a < 0.84375) {
    const z = a * a
    return sign * (a * (1 + polyEval(ERF_COEFF.pp, z) / polyEval(ERF_COEFF.qq, z)))
  }
  if (a < 1.25) {
    const s = a - 1
    return sign * (ERF_COEFF.erx + polyEval(ERF_COEFF.pa, s) / polyEval(ERF_COEFF.qa, s))
  }
  if (a < 1 / 0.35) {
    const z = a * a
    const s = 1 / z
    return sign * (1 - Math.exp(-z - 0.5625 + polyEval(ERF_COEFF.ra, s) / polyEval(ERF_COEFF.sa, s)) / a)
  }
  {
    const z = a * a
    const s = 1 / z
    return sign * (1 - Math.exp(-z - 0.5625 + polyEval(ERF_COEFF.rb, s) / polyEval(ERF_COEFF.sb, s)) / a)
  }
}

export function ndtr(x) {
  const v = 0.5 + 0.5 * erfScalar(x / SQRT2)
  if (v <= 0) return 0
  if (v >= 1) return 1
  return v
}

export function logNdtr(x) {
  if (x > 6) {
    return -ndtr(-x)
  }
  if (x > -20) {
    return Math.log(ndtr(x))
  }

  const logLhs = -0.5 * x * x - Math.log(-x) - LOG_SQRT_2PI
  let lastTotal = 0
  let rhs = 1
  let numerator = 1
  let denomFactor = 1
  const denomCons = 1 / (x * x)
  let sign = 1
  let i = 0

  while (Math.abs(lastTotal - rhs) > Number.EPSILON) {
    i += 1
    lastTotal = rhs
    sign = -sign
    denomFactor *= denomCons
    numerator *= 2 * i - 1
    rhs += sign * numerator * denomFactor
  }

  return logLhs + Math.log(rhs)
}

export function logSum(logP, logQ) {
  const a = Math.max(logP, logQ)
  const b = Math.min(logP, logQ)
  return a + Math.log1p(Math.exp(b - a))
}

export function logDiff(logP, logQ) {
  if (logQ >= logP) {
    return -Infinity
  }
  return logP + Math.log1p(-Math.exp(logQ - logP))
}

export function logGaussMass(a, b) {
  if (b <= 0) {
    return logDiff(logNdtr(b), logNdtr(a))
  }
  if (a > 0) {
    return logGaussMass(-b, -a)
  }
  const central = 1 - ndtr(a) - ndtr(-b)
  if (central > 0) {
    return Math.log(central)
  }
  const leftMass = logDiff(logNdtr(0), logNdtr(a))
  const rightMass = logDiff(logNdtr(b), logNdtr(0))
  return logSum(leftMass, rightMass)
}

export function ndtriExp(y) {
  let flipped = false
  let z = y
  if (y > -1e-2) {
    flipped = true
    z = Math.log(-Math.expm1(y))
  }

  let x
  if (z < -5) {
    x = -Math.sqrt(-2 * (z + LOG_SQRT_2PI))
  } else {
    x = -NDTRI_EXP_APPROX_C * Math.log(Math.expm1(-z))
  }

  for (let i = 0; i < 100; i += 1) {
    const logNdtrX = logNdtr(x)
    const logNormPdfX = -0.5 * x * x - LOG_SQRT_2PI
    const dx = (logNdtrX - z) * Math.exp(logNdtrX - logNormPdfX)
    x -= dx
    if (Math.abs(dx) < 1e-8 * Math.abs(x)) {
      break
    }
  }

  if (flipped) {
    x *= -1
  }

  return x
}

export function truncnormPpf(q, a, b) {
  if (q === 0) return a
  if (q === 1) return b
  if (a === b) return Number.NaN

  const logMass = logGaussMass(a, b)
  if (a < 0) {
    const logPhiX = logSum(logNdtr(a), Math.log(q) + logMass)
    return ndtriExp(logPhiX)
  }

  const logPhiX = logSum(logNdtr(-b), Math.log1p(-q) + logMass)
  return -ndtriExp(logPhiX)
}

export function truncnormLogpdf(x, a, b, loc, scale) {
  const xn = (x - loc) / scale
  const out = -0.5 * xn * xn - LOG_SQRT_2PI - logGaussMass(a, b) - Math.log(scale)
  if (a === b) return Number.NaN
  if (xn < a || xn > b) return -Infinity
  return out
}
