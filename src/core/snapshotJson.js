import { SPECIAL_NUMBER_MARKER } from './constants.js'
import { isPlainObject } from './objectUtils.js'

export function encodeNumberForSnapshot(value) {
  if (Number.isNaN(value)) {
    return { [SPECIAL_NUMBER_MARKER]: 'NaN' }
  }
  if (value === Infinity) {
    return { [SPECIAL_NUMBER_MARKER]: '+Infinity' }
  }
  if (value === -Infinity) {
    return { [SPECIAL_NUMBER_MARKER]: '-Infinity' }
  }
  if (Object.is(value, -0)) {
    return { [SPECIAL_NUMBER_MARKER]: '-0' }
  }
  return value
}

export function decodeNumberFromSnapshot(value) {
  if (
    !isPlainObject(value) ||
    !(SPECIAL_NUMBER_MARKER in value) ||
    Object.keys(value).length !== 1
  ) {
    return value
  }
  const token = value[SPECIAL_NUMBER_MARKER]
  if (token === 'NaN') return Number.NaN
  if (token === '+Infinity') return Infinity
  if (token === '-Infinity') return -Infinity
  if (token === '-0') return -0
  throw new Error(`Unknown serialized number token: ${token}`)
}

export function serializeJsonValueForSnapshot(value) {
  if (value === null) {
    return null
  }

  const valueType = typeof value
  if (valueType === 'number') {
    return encodeNumberForSnapshot(value)
  }
  if (valueType === 'string' || valueType === 'boolean') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => serializeJsonValueForSnapshot(item))
  }
  if (isPlainObject(value)) {
    const out = {}
    for (const [key, item] of Object.entries(value)) {
      out[key] = serializeJsonValueForSnapshot(item)
    }
    return out
  }

  throw new Error(`Cannot serialize value of type "${valueType}" in study snapshot.`)
}

export function deserializeJsonValueFromSnapshot(value) {
  if (value === null) {
    return null
  }
  if (typeof value !== 'object') {
    return value
  }
  if (Array.isArray(value)) {
    return value.map((item) => deserializeJsonValueFromSnapshot(item))
  }
  if (SPECIAL_NUMBER_MARKER in value && Object.keys(value).length === 1) {
    return decodeNumberFromSnapshot(value)
  }
  const out = {}
  for (const [key, item] of Object.entries(value)) {
    out[key] = deserializeJsonValueFromSnapshot(item)
  }
  return out
}

export function cloneJsonValue(value) {
  return deserializeJsonValueFromSnapshot(serializeJsonValueForSnapshot(value))
}
