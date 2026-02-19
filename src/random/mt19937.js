export function randomChoiceIndicesByWeights(rng, probabilities, size) {
  const cumulative = new Array(probabilities.length)
  let running = 0
  for (let i = 0; i < probabilities.length; i += 1) {
    running += probabilities[i]
    cumulative[i] = running
  }
  cumulative[cumulative.length - 1] = 1

  const out = new Array(size)
  for (let i = 0; i < size; i += 1) {
    const r = rng.randomSample()
    let idx = 0
    while (idx < cumulative.length && r >= cumulative[idx]) {
      idx += 1
    }
    out[i] = idx
  }
  return out
}

export class MT19937 {
  constructor(seed = null) {
    this.N = 624
    this.M = 397
    this.MATRIX_A = 0x9908b0df
    this.UPPER_MASK = 0x80000000
    this.LOWER_MASK = 0x7fffffff
    this.mt = new Uint32Array(this.N)
    this.mti = this.N + 1
    if (seed !== null && seed !== undefined) {
      this.seed(seed)
    } else {
      this.seed((Date.now() >>> 0) ^ 0x6c078965)
    }
  }

  seed(seed) {
    let s = Number(seed) >>> 0
    this.mt[0] = s
    for (this.mti = 1; this.mti < this.N; this.mti += 1) {
      s = this.mt[this.mti - 1] ^ (this.mt[this.mti - 1] >>> 30)
      this.mt[this.mti] =
        ((((s & 0xffff0000) >>> 16) * 1812433253) << 16) +
        (s & 0x0000ffff) * 1812433253 +
        this.mti
      this.mt[this.mti] >>>= 0
    }
  }

  _genInt32() {
    let y
    const mag01 = [0x0, this.MATRIX_A]

    if (this.mti >= this.N) {
      let kk
      for (kk = 0; kk < this.N - this.M; kk += 1) {
        y = (this.mt[kk] & this.UPPER_MASK) | (this.mt[kk + 1] & this.LOWER_MASK)
        this.mt[kk] = this.mt[kk + this.M] ^ (y >>> 1) ^ mag01[y & 0x1]
      }
      for (; kk < this.N - 1; kk += 1) {
        y = (this.mt[kk] & this.UPPER_MASK) | (this.mt[kk + 1] & this.LOWER_MASK)
        this.mt[kk] = this.mt[kk + (this.M - this.N)] ^ (y >>> 1) ^ mag01[y & 0x1]
      }
      y = (this.mt[this.N - 1] & this.UPPER_MASK) | (this.mt[0] & this.LOWER_MASK)
      this.mt[this.N - 1] = this.mt[this.M - 1] ^ (y >>> 1) ^ mag01[y & 0x1]
      this.mti = 0
    }

    y = this.mt[this.mti]
    this.mti += 1

    y ^= y >>> 11
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= y >>> 18

    return y >>> 0
  }

  randomSample() {
    const a = this._genInt32() >>> 5
    const b = this._genInt32() >>> 6
    return (a * 67108864 + b) / 9007199254740992
  }

  rand(size) {
    const out = new Array(size)
    for (let i = 0; i < size; i += 1) {
      out[i] = this.randomSample()
    }
    return out
  }

  uniform(low, high) {
    return low + (high - low) * this.randomSample()
  }

  choiceWeighted(probabilities, size) {
    return randomChoiceIndicesByWeights(this, probabilities, size)
  }
}
