import { getMsb } from '../core/numberUtils.js'

export function heapSortArgSegment(values, indices, start, end) {
  const less = (i, j) => values[indices[i]] < values[indices[j]]
  const swap = (i, j) => {
    const tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp
  }
  const length = end - start + 1
  const siftDown = (root, size) => {
    while (true) {
      let child = root * 2 + 1
      if (child >= size) return
      if (child + 1 < size && less(start + child, start + child + 1)) {
        child += 1
      }
      if (!less(start + root, start + child)) {
        return
      }
      swap(start + root, start + child)
      root = child
    }
  }
  for (let i = Math.floor(length / 2) - 1; i >= 0; i -= 1) {
    siftDown(i, length)
  }
  for (let i = length - 1; i > 0; i -= 1) {
    swap(start, start + i)
    siftDown(0, i)
  }
}

export function numpyQuickArgSort(values) {
  const num = values.length
  const tosort = Array.from({ length: num }, (_, i) => i)
  if (num <= 1) {
    return tosort
  }

  const SMALL_QUICKSORT = 15
  const stack = []
  let pl = 0
  let pr = num - 1
  let cdepth = getMsb(num) * 2

  const lessValue = (idxA, idxB) => values[tosort[idxA]] < values[tosort[idxB]]
  const lessPivot = (value, idxB) => value < values[tosort[idxB]]
  const lessFrom = (idxA, value) => values[tosort[idxA]] < value
  const swap = (a, b) => {
    const tmp = tosort[a]
    tosort[a] = tosort[b]
    tosort[b] = tmp
  }

  for (;;) {
    if (cdepth < 0) {
      heapSortArgSegment(values, tosort, pl, pr)
      if (stack.length === 0) {
        break
      }
      const last = stack.pop()
      pl = last.pl
      pr = last.pr
      cdepth = last.cdepth
      continue
    }

    while (pr - pl > SMALL_QUICKSORT) {
      let pm = pl + ((pr - pl) >> 1)
      if (lessValue(pm, pl)) {
        swap(pm, pl)
      }
      if (lessValue(pr, pm)) {
        swap(pr, pm)
      }
      if (lessValue(pm, pl)) {
        swap(pm, pl)
      }

      const vp = values[tosort[pm]]
      let pi = pl
      let pj = pr - 1
      swap(pm, pj)

      for (;;) {
        do {
          pi += 1
        } while (lessFrom(pi, vp))
        do {
          pj -= 1
        } while (lessPivot(vp, pj))
        if (pi >= pj) {
          break
        }
        swap(pi, pj)
      }

      swap(pi, pr - 1)

      cdepth -= 1
      if (pi - pl < pr - pi) {
        stack.push({ pl: pi + 1, pr, cdepth })
        pr = pi - 1
      } else {
        stack.push({ pl, pr: pi - 1, cdepth })
        pl = pi + 1
      }
    }

    for (let pi = pl + 1; pi <= pr; pi += 1) {
      const vi = tosort[pi]
      const vp = values[vi]
      let pj = pi
      let pk = pi - 1
      while (pj > pl && vp < values[tosort[pk]]) {
        tosort[pj] = tosort[pk]
        pj -= 1
        pk -= 1
      }
      tosort[pj] = vi
    }

    if (stack.length === 0) {
      break
    }

    const last = stack.pop()
    pl = last.pl
    pr = last.pr
    cdepth = last.cdepth
  }

  return tosort
}
