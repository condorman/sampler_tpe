import {
  CategoricalDistribution,
  FloatDistribution,
  IntDistribution
} from '../internal/impl.js'
import { internals } from '../internal/impl.js'

export { FloatDistribution, IntDistribution, CategoricalDistribution }
export const distributionContainsValue = internals.distributionContainsValue
