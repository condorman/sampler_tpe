export function compareRowsLex(a, b) {
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] < b[i]) return -1
    if (a[i] > b[i]) return 1
  }
  return 0
}

export function uniqueSortedRowsWithInverse(rows) {
  const order = rows.map((_, i) => i).sort((ia, ib) => compareRowsLex(rows[ia], rows[ib]))
  const uniqueRows = []
  const inverse = new Array(rows.length)

  for (let i = 0; i < order.length; i += 1) {
    const idx = order[i]
    const row = rows[idx]
    if (uniqueRows.length === 0 || compareRowsLex(row, uniqueRows[uniqueRows.length - 1]) !== 0) {
      uniqueRows.push(row.slice())
    }
    inverse[idx] = uniqueRows.length - 1
  }

  return { uniqueRows, inverse }
}

export function isParetoFront2D(uniqueLexsortedLossValues) {
  const n = uniqueLexsortedLossValues.length
  if (n === 0) return []
  const onFront = new Array(n).fill(true)
  let currentMin = uniqueLexsortedLossValues[0][1]
  for (let i = 1; i < n; i += 1) {
    const v = uniqueLexsortedLossValues[i][1]
    onFront[i] = v < currentMin
    if (v < currentMin) {
      currentMin = v
    }
  }
  return onFront
}

export function isParetoFrontND(uniqueLexsortedLossValues) {
  const lossValues = uniqueLexsortedLossValues.map((row) => row.slice(1))
  const n = lossValues.length
  const onFront = new Array(n).fill(false)
  let remainingIndices = Array.from({ length: n }, (_, i) => i)

  while (remainingIndices.length > 0) {
    const newNonDominatedIndex = remainingIndices[0]
    onFront[newNonDominatedIndex] = true
    const nextRemaining = []
    for (let r = 0; r < remainingIndices.length; r += 1) {
      const idx = remainingIndices[r]
      let anyLess = false
      for (let k = 0; k < lossValues[idx].length; k += 1) {
        if (lossValues[idx][k] < lossValues[newNonDominatedIndex][k]) {
          anyLess = true
          break
        }
      }
      if (anyLess) {
        nextRemaining.push(idx)
      }
    }
    remainingIndices = nextRemaining
  }

  return onFront
}

export function isParetoFrontForUniqueSorted(uniqueLexsortedLossValues) {
  const nObjectives = uniqueLexsortedLossValues[0].length
  if (nObjectives === 1) {
    const onFront = new Array(uniqueLexsortedLossValues.length).fill(false)
    onFront[0] = true
    return onFront
  }
  if (nObjectives === 2) {
    return isParetoFront2D(uniqueLexsortedLossValues)
  }
  return isParetoFrontND(uniqueLexsortedLossValues)
}

export function isParetoFront(lossValues, assumeUniqueLexsorted = false) {
  if (lossValues.length === 0) return []
  if (assumeUniqueLexsorted) {
    return isParetoFrontForUniqueSorted(lossValues)
  }
  const { uniqueRows, inverse } = uniqueSortedRowsWithInverse(lossValues)
  const uniqueOnFront = isParetoFrontForUniqueSorted(uniqueRows)
  return inverse.map((i) => uniqueOnFront[i])
}

export function calculateNondominationRank(lossValues, nBelow = null) {
  if (lossValues.length === 0 || (nBelow !== null && nBelow <= 0)) {
    return new Array(lossValues.length).fill(0)
  }

  const nObjectives = lossValues[0].length
  if (nObjectives === 1) {
    const values = lossValues.map((row) => row[0])
    const sortedUnique = [...new Set(values.slice().sort((a, b) => a - b))]
    const rankMap = new Map()
    for (let i = 0; i < sortedUnique.length; i += 1) {
      rankMap.set(sortedUnique[i], i)
    }
    return values.map((v) => rankMap.get(v))
  }

  const { uniqueRows, inverse } = uniqueSortedRowsWithInverse(lossValues)
  let remaining = uniqueRows.map((row, i) => ({ row, idx: i }))
  const nUnique = uniqueRows.length
  const clippedNBelow = Math.min(nBelow === null ? nUnique : nBelow, nUnique)
  const ranks = new Array(nUnique).fill(0)
  let rank = 0

  while (nUnique - remaining.length < clippedNBelow) {
    const rows = remaining.map((it) => it.row)
    const onFront = isParetoFront(rows, true)
    const next = []
    for (let i = 0; i < remaining.length; i += 1) {
      if (onFront[i]) {
        ranks[remaining[i].idx] = rank
      } else {
        next.push(remaining[i])
      }
    }
    remaining = next
    rank += 1
  }

  for (const r of remaining) {
    ranks[r.idx] = rank
  }

  return inverse.map((i) => ranks[i])
}

export function fastNonDominationRank(lossValues, penalty = null, nBelow = null) {
  if (lossValues.length === 0) {
    return []
  }

  let nBelowLocal = nBelow === null ? lossValues.length : nBelow
  if (nBelowLocal <= 0) {
    throw new Error('nBelow must be positive.')
  }

  if (penalty === null) {
    return calculateNondominationRank(lossValues, nBelowLocal)
  }

  if (penalty.length !== lossValues.length) {
    throw new Error('penalty and lossValues length mismatch.')
  }

  const ranks = new Array(lossValues.length).fill(-1)

  const feasibleIndices = []
  const infeasibleIndices = []
  const nanIndices = []

  for (let i = 0; i < penalty.length; i += 1) {
    const p = penalty[i]
    if (Number.isNaN(p)) {
      nanIndices.push(i)
    } else if (p <= 0) {
      feasibleIndices.push(i)
    } else {
      infeasibleIndices.push(i)
    }
  }

  const feasibleLosses = feasibleIndices.map((i) => lossValues[i])
  const feasibleRanks = calculateNondominationRank(feasibleLosses, nBelowLocal)
  for (let i = 0; i < feasibleIndices.length; i += 1) {
    ranks[feasibleIndices[i]] = feasibleRanks[i]
  }
  nBelowLocal -= feasibleIndices.length

  const topRankInfeasible = Math.max(-1, ...feasibleIndices.map((i) => ranks[i])) + 1
  const infeasiblePenaltyAsLoss = infeasibleIndices.map((i) => [penalty[i]])
  const infeasibleRanks = calculateNondominationRank(infeasiblePenaltyAsLoss, nBelowLocal)
  for (let i = 0; i < infeasibleIndices.length; i += 1) {
    ranks[infeasibleIndices[i]] = topRankInfeasible + infeasibleRanks[i]
  }
  nBelowLocal -= infeasibleIndices.length

  const topRankPenaltyNan =
    Math.max(-1, ...ranks.filter((r, i) => !Number.isNaN(penalty[i]))) + 1
  const nanLosses = nanIndices.map((i) => lossValues[i])
  const nanRanks = calculateNondominationRank(nanLosses, nBelowLocal)
  for (let i = 0; i < nanIndices.length; i += 1) {
    ranks[nanIndices[i]] = topRankPenaltyNan + nanRanks[i]
  }

  return ranks
}
