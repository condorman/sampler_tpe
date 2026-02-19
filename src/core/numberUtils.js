export function clip(value, low, high) {
  if (value < low) return low
  if (value > high) return high
  return value
}

export function roundToNearestEven(value) {
  const floorValue = Math.floor(value)
  const fraction = value - floorValue
  const eps = 1e-12
  if (fraction < 0.5 - eps) {
    return floorValue
  }
  if (fraction > 0.5 + eps) {
    return floorValue + 1
  }
  return floorValue % 2 === 0 ? floorValue : floorValue + 1
}

export function nextafter(x, y) {
  if (Number.isNaN(x) || Number.isNaN(y)) return Number.NaN
  if (x === y) return y
  if (!Number.isFinite(x)) return x
  if (x === 0) {
    return y > 0 ? Number.MIN_VALUE : -Number.MIN_VALUE
  }

  const buffer = new ArrayBuffer(8)
  const view = new DataView(buffer)
  view.setFloat64(0, x, false)
  const hi = BigInt(view.getUint32(0, false))
  const lo = BigInt(view.getUint32(4, false))
  let bits = (hi << 32n) | lo

  const increment = (y > x) === (x > 0)
  bits = increment ? bits + 1n : bits - 1n

  view.setUint32(0, Number((bits >> 32n) & 0xffffffffn), false)
  view.setUint32(4, Number(bits & 0xffffffffn), false)
  return view.getFloat64(0, false)
}

export function getMsb(n) {
  let msb = -1
  while (n > 0) {
    n >>= 1
    msb += 1
  }
  return msb
}
