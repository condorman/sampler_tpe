export function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

export function shallowCopy(obj) {
  return Object.assign({}, obj)
}

export function sortObjectEntries(obj) {
  return Object.keys(obj)
    .sort()
    .map((key) => [key, obj[key]])
}

export function hasOwn(obj, key) {
  return Object.prototype.hasOwnProperty.call(obj, key)
}
