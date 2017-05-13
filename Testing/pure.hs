{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
import           Control.Category ((>>>))
import           Control.Monad.Trans
import           Data.Functor.Identity
import           GHC.TypeLits
import           Test.QuickCheck hiding (scale)
import           Test.QuickCheck.Monadic

import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Flat.Flat
import           AI.Funn.Space

import Testing.Util

prop_fcdiff :: Property
prop_fcdiff = checkGradientI (fcDiff @10 @6)

prop_sumdiff :: Property
prop_sumdiff = checkGradientI (sumDiff @10)

prop_preludiff :: Property
prop_preludiff = checkGradientI (preluDiff @10)

prop_sigmoiddiff :: Property
prop_sigmoiddiff = checkGradientI (sigmoidDiff @10)

prop_tanhdiff :: Property
prop_tanhdiff = checkGradientI (tanhDiff @10)

prop_mergediff :: Property
prop_mergediff = checkGradientI (mergeDiff @Identity @10 @13)

prop_splitdiff :: Property
prop_splitdiff = checkGradientI (splitDiff @Identity @4 @5)

prop_quadraticcost :: Property
prop_quadraticcost = checkGradientI (quadraticCost @Identity @10)

prop_softmaxcost :: Property
prop_softmaxcost = checkGradientI (putR 0 >>> softmaxCost @Identity @3)


-- Make TemplateHaskell aware of above definitions.
$(return [])

main :: IO Bool
main = $(quickCheckAll)
