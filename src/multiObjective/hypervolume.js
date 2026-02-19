import { isParetoFront, uniqueSortedRowsWithInverse } from './pareto.js'

export function compute2dHypervolume(sortedParetoSols, referencePoint) {
  let hv = 0
  for (let i = 0; i < sortedParetoSols.length; i += 1) {
    const rectDiagY = i === 0 ? referencePoint[1] : sortedParetoSols[i - 1][1]
    const edgeX = referencePoint[0] - sortedParetoSols[i][0]
    const edgeY = rectDiagY - sortedParetoSols[i][1]
    hv += edgeX * edgeY
  }
  return hv
}

export function compute3dHypervolume(sortedParetoSols, referencePoint) {
  const n = sortedParetoSols.length
  const yOrder = Array.from({ length: n }, (_, i) => i).sort(
    (a, b) => sortedParetoSols[a][1] - sortedParetoSols[b][1]
  )

  const zDelta = Array.from({ length: n }, () => new Array(n).fill(0))
  for (let j = 0; j < n; j += 1) {
    const row = yOrder[j]
    zDelta[row][j] = referencePoint[2] - sortedParetoSols[row][2]
  }

  for (let i = 0; i < n; i += 1) {
    for (let j = 1; j < n; j += 1) {
      if (zDelta[i][j] < zDelta[i][j - 1]) {
        zDelta[i][j] = zDelta[i][j - 1]
      }
    }
  }
  for (let j = 0; j < n; j += 1) {
    for (let i = 1; i < n; i += 1) {
      if (zDelta[i][j] < zDelta[i - 1][j]) {
        zDelta[i][j] = zDelta[i - 1][j]
      }
    }
  }

  const xVals = sortedParetoSols.map((row) => row[0])
  const yVals = yOrder.map((idx) => sortedParetoSols[idx][1])

  const xDelta = new Array(n)
  const yDelta = new Array(n)
  for (let i = 0; i < n; i += 1) {
    xDelta[i] = (i + 1 < n ? xVals[i + 1] : referencePoint[0]) - xVals[i]
    yDelta[i] = (i + 1 < n ? yVals[i + 1] : referencePoint[1]) - yVals[i]
  }

  let hv = 0
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      hv += zDelta[j][i] * yDelta[i] * xDelta[j]
    }
  }

  return hv
}

export function computeHvRecursive(sortedLossVals, referencePoint) {
  const n = sortedLossVals.length
  if (n === 1) {
    let hv = 1
    for (let i = 0; i < referencePoint.length; i += 1) {
      hv *= referencePoint[i] - sortedLossVals[0][i]
    }
    return hv
  }

  if (n === 2) {
    let hv1 = 1
    let hv2 = 1
    let inter = 1
    for (let i = 0; i < referencePoint.length; i += 1) {
      hv1 *= referencePoint[i] - sortedLossVals[0][i]
      hv2 *= referencePoint[i] - sortedLossVals[1][i]
      inter *= referencePoint[i] - Math.max(sortedLossVals[0][i], sortedLossVals[1][i])
    }
    return hv1 + hv2 - inter
  }

  const inclusiveHvs = sortedLossVals.map((row) => {
    let p = 1
    for (let i = 0; i < row.length; i += 1) {
      p *= referencePoint[i] - row[i]
    }
    return p
  })

  const limitedSolsArray = sortedLossVals.map((rowI) =>
    sortedLossVals.map((rowJ) => rowI.map((v, dim) => Math.max(v, rowJ[dim])))
  )

  let hv = inclusiveHvs[inclusiveHvs.length - 1]
  for (let i = 0; i < inclusiveHvs.length - 1; i += 1) {
    const limited = limitedSolsArray[i].slice(i + 1)
    hv += computeExclusiveHv(limited, inclusiveHvs[i], referencePoint)
  }
  return hv
}

export function computeExclusiveHv(limitedSols, inclusiveHv, referencePoint) {
  if (limitedSols.length <= 3) {
    return inclusiveHv - computeHvRecursive(limitedSols, referencePoint)
  }
  const onFront = isParetoFront(limitedSols, true)
  const front = []
  for (let i = 0; i < limitedSols.length; i += 1) {
    if (onFront[i]) {
      front.push(limitedSols[i])
    }
  }
  return inclusiveHv - computeHvRecursive(front, referencePoint)
}

export function computeHypervolume(lossVals, referencePoint, assumePareto = false) {
  for (let i = 0; i < lossVals.length; i += 1) {
    for (let d = 0; d < referencePoint.length; d += 1) {
      if (lossVals[i][d] > referencePoint[d]) {
        throw new Error('All points must dominate or equal reference point.')
      }
    }
  }
  for (let i = 0; i < referencePoint.length; i += 1) {
    if (!Number.isFinite(referencePoint[i])) {
      return Infinity
    }
  }
  if (lossVals.length === 0) {
    return 0
  }

  let sortedParetoSols
  if (!assumePareto) {
    const { uniqueRows } = uniqueSortedRowsWithInverse(lossVals)
    const onFront = isParetoFront(uniqueRows, true)
    sortedParetoSols = uniqueRows.filter((_, i) => onFront[i])
  } else {
    sortedParetoSols = [...lossVals].sort((a, b) => a[0] - b[0])
  }

  let hv
  if (referencePoint.length === 2) {
    hv = compute2dHypervolume(sortedParetoSols, referencePoint)
  } else if (referencePoint.length === 3) {
    hv = compute3dHypervolume(sortedParetoSols, referencePoint)
  } else {
    hv = computeHvRecursive(sortedParetoSols, referencePoint)
  }

  return Number.isFinite(hv) ? hv : Infinity
}
