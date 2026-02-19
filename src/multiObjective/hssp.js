import { EPS } from '../core/constants.js'
import { isParetoFront, uniqueSortedRowsWithInverse } from './pareto.js'
import { computeHypervolume } from './hypervolume.js'

export function solveHssp2d(rankLossVals, rankIndices, subsetSize, referencePoint) {
  const nTrials = rankLossVals.length
  let sortedIndices = Array.from({ length: nTrials }, (_, i) => i)
  let sortedLossVals = rankLossVals.map((row) => row.slice())
  let rectDiags = Array.from({ length: nTrials }, () => referencePoint.slice())
  const selectedIndices = new Array(subsetSize)

  for (let i = 0; i < subsetSize; i += 1) {
    let maxIndex = 0
    let maxContrib = -Infinity
    for (let j = 0; j < sortedLossVals.length; j += 1) {
      const contrib =
        (rectDiags[j][0] - sortedLossVals[j][0]) *
        (rectDiags[j][1] - sortedLossVals[j][1])
      if (contrib > maxContrib) {
        maxContrib = contrib
        maxIndex = j
      }
    }

    selectedIndices[i] = rankIndices[sortedIndices[maxIndex]]
    const loss = sortedLossVals[maxIndex].slice()

    const keep = []
    for (let j = 0; j < sortedLossVals.length; j += 1) {
      if (j !== maxIndex) keep.push(j)
    }

    const nextSortedIndices = []
    const nextRectDiags = []
    const nextLossVals = []
    for (let j = 0; j < keep.length; j += 1) {
      const source = keep[j]
      nextSortedIndices.push(sortedIndices[source])
      nextRectDiags.push(rectDiags[source].slice())
      nextLossVals.push(sortedLossVals[source].slice())
    }

    for (let j = 0; j < nextRectDiags.length; j += 1) {
      if (j < maxIndex) {
        nextRectDiags[j][0] = Math.min(loss[0], nextRectDiags[j][0])
      } else {
        nextRectDiags[j][1] = Math.min(loss[1], nextRectDiags[j][1])
      }
    }

    sortedIndices = nextSortedIndices
    rectDiags = nextRectDiags
    sortedLossVals = nextLossVals
  }

  return selectedIndices
}

export function lazyContribsUpdate(
  contribs,
  paretoLossValues,
  selectedVecs,
  referencePoint,
  hvSelected
) {
  if (!Number.isFinite(hvSelected)) {
    return new Array(contribs.length).fill(Infinity)
  }

  const inclusiveHvs = paretoLossValues.map((row) => {
    let p = 1
    for (let i = 0; i < row.length; i += 1) {
      p *= referencePoint[i] - row[i]
    }
    return p
  })

  const intersec = paretoLossValues.map((row) =>
    selectedVecs.slice(0, -1).map((sel) => row.map((v, d) => Math.max(v, sel[d])))
  )

  const updated = [...contribs]
  for (let i = 0; i < updated.length; i += 1) {
    const latestIntersection = paretoLossValues[i].map((v, d) =>
      Math.max(v, selectedVecs[selectedVecs.length - 1][d])
    )
    let latestVolume = 1
    for (let d = 0; d < latestIntersection.length; d += 1) {
      latestVolume *= referencePoint[d] - latestIntersection[d]
    }
    updated[i] = Math.min(updated[i], inclusiveHvs[i] - latestVolume)
  }

  let maxContrib = 0
  const isHvCalcFast = paretoLossValues[0].length <= 3
  const order = Array.from({ length: updated.length }, (_, i) => i).sort(
    (a, b) => updated[b] - updated[a]
  )

  for (const i of order) {
    if (!Number.isFinite(inclusiveHvs[i])) {
      maxContrib = updated[i] = Infinity
      continue
    }
    if (updated[i] < maxContrib) {
      continue
    }

    if (isHvCalcFast) {
      const plusSet = [...selectedVecs]
      plusSet[plusSet.length - 1] = paretoLossValues[i].slice()
      const hvPlus = computeHypervolume(plusSet, referencePoint, true)
      updated[i] = hvPlus - hvSelected
    } else {
      updated[i] = inclusiveHvs[i] - computeHypervolume(intersec[i], referencePoint)
    }

    if (updated[i] > maxContrib) {
      maxContrib = updated[i]
    }
  }

  return updated
}

export function solveHsspOnUniqueLossVals(rankLossVals, rankIndices, subsetSize, referencePoint) {
  if (!referencePoint.every((v) => Number.isFinite(v))) {
    return rankIndices.slice(0, subsetSize)
  }
  if (rankIndices.length === subsetSize) {
    return rankIndices.slice()
  }
  if (rankLossVals[0].length === 2) {
    return solveHssp2d(rankLossVals, rankIndices, subsetSize, referencePoint)
  }

  let contribs = rankLossVals.map((row) => {
    let p = 1
    for (let i = 0; i < row.length; i += 1) {
      p *= referencePoint[i] - row[i]
    }
    return p
  })
  const selectedIndices = []
  const selectedVecs = []
  let indices = Array.from({ length: rankLossVals.length }, (_, i) => i)
  let lossVals = rankLossVals.map((row) => row.slice())
  let hv = 0

  for (let k = 0; k < subsetSize; k += 1) {
    let maxIndex = 0
    for (let i = 1; i < contribs.length; i += 1) {
      if (contribs[i] > contribs[maxIndex]) {
        maxIndex = i
      }
    }

    hv += contribs[maxIndex]
    selectedIndices.push(indices[maxIndex])
    selectedVecs.push(lossVals[maxIndex].slice())

    if (k === subsetSize - 1) {
      break
    }

    const keep = []
    for (let i = 0; i < contribs.length; i += 1) {
      if (i !== maxIndex) keep.push(i)
    }

    contribs = keep.map((i) => contribs[i])
    indices = keep.map((i) => indices[i])
    lossVals = keep.map((i) => lossVals[i])

    const selectedForUpdate = [...selectedVecs, new Array(referencePoint.length).fill(0)]
    contribs = lazyContribsUpdate(contribs, lossVals, selectedForUpdate, referencePoint, hv)
  }

  return selectedIndices.map((i) => rankIndices[i])
}

export function solveHssp(rankLossVals, rankIndices, subsetSize, referencePoint) {
  if (subsetSize === rankIndices.length) {
    return rankIndices.slice()
  }

  const { uniqueRows, inverse } = uniqueSortedRowsWithInverse(rankLossVals)
  const firstOccurrence = new Array(uniqueRows.length).fill(-1)
  for (let i = 0; i < inverse.length; i += 1) {
    const u = inverse[i]
    if (firstOccurrence[u] === -1) {
      firstOccurrence[u] = i
    }
  }

  if (uniqueRows.length < subsetSize) {
    const chosen = new Array(rankIndices.length).fill(false)
    for (const idx of firstOccurrence) chosen[idx] = true
    const duplicated = []
    for (let i = 0; i < chosen.length; i += 1) {
      if (!chosen[i]) duplicated.push(i)
    }
    for (let i = 0; i < subsetSize - uniqueRows.length; i += 1) {
      chosen[duplicated[i]] = true
    }
    const out = []
    for (let i = 0; i < chosen.length; i += 1) {
      if (chosen[i]) out.push(rankIndices[i])
    }
    return out
  }

  const selectedUnique = solveHsspOnUniqueLossVals(
    uniqueRows,
    firstOccurrence,
    subsetSize,
    referencePoint
  )
  return selectedUnique.map((i) => rankIndices[i])
}

export function getReferencePoint(lossVals) {
  const dims = lossVals[0].length
  const worst = new Array(dims).fill(-Infinity)
  for (const row of lossVals) {
    for (let d = 0; d < dims; d += 1) {
      if (row[d] > worst[d]) worst[d] = row[d]
    }
  }

  const ref = worst.map((w) => Math.max(1.1 * w, 0.9 * w))
  for (let i = 0; i < ref.length; i += 1) {
    if (ref[i] === 0) ref[i] = EPS
  }
  return ref
}
