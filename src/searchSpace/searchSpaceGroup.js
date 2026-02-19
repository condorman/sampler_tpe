import { shallowCopy } from '../core/objectUtils.js'

export class SearchSpaceGroup {
  constructor() {
    this.searchSpaces = []
  }

  addDistributions(distributions) {
    let distKeys = new Set(Object.keys(distributions))
    const nextSpaces = []

    for (const searchSpace of this.searchSpaces) {
      const keys = new Set(Object.keys(searchSpace))
      const intersect = {}
      const left = {}
      for (const key of keys) {
        if (distKeys.has(key)) {
          intersect[key] = searchSpace[key]
        } else {
          left[key] = searchSpace[key]
        }
      }
      if (Object.keys(intersect).length > 0) nextSpaces.push(intersect)
      if (Object.keys(left).length > 0) nextSpaces.push(left)
      for (const key of keys) {
        distKeys.delete(key)
      }
    }

    const right = {}
    for (const key of distKeys) {
      right[key] = distributions[key]
    }
    if (Object.keys(right).length > 0) nextSpaces.push(right)
    this.searchSpaces = nextSpaces
  }

  clone() {
    const c = new SearchSpaceGroup()
    c.searchSpaces = this.searchSpaces.map((sp) => shallowCopy(sp))
    return c
  }
}
