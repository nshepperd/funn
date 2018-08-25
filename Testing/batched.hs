{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}

import           Control.Category ((>>>), id)
import           Control.Monad.Trans
import           Data.Functor.Identity
import           Data.Proxy
import           GHC.TypeLits
import           Prelude hiding (id)
import           Test.QuickCheck hiding (scale)

import           AI.Funn.CL.Batched.GLOW
import           AI.Funn.CL.Batched.Layers.FullyConnected
import           AI.Funn.CL.Batched.Layers.GLOW
import           AI.Funn.CL.Batched.Layers.Simple
import           AI.Funn.CL.Batched.Layers.Triangular
import           AI.Funn.CL.Batched.Network (Network(..), runNetwork)
import           AI.Funn.CL.Blob (Blob(..))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor)
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..), runDiffForward)
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as C
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Space

import           Testing.Util


checkGradientNetB :: forall a b n1 n2 p.
                     (Loadable a n1, D a ~ a,
                      Loadable b n2, D b ~ b,
                      KnownNat p)
                  => Network IO 1 p a b -> Property
checkGradientNetB net = checkGradientCL (Diff run :: Diff IO (Tensor '[p], a) b)
  where
    run (par, a) = do
      (b, k) <- runNetwork net (par, a)
      return (b, backward k)
    backward k db = do
      (dpar :: Tensor '[1, p], da) <- k db
      return (Tensor.reshape dpar :: Tensor '[p], da)

checkGradientInvB :: forall a b n1 n2 p.
                     (Loadable a n1, D a ~ a,
                      Loadable b n2, D b ~ b,
                      KnownNat p)
                  => Invertible IO 1 p a b -> Property
checkGradientInvB net = checkGradientNetB (invForward net) .&&. checkGradientNetB (invBackward net)

-- Layers

prop_fc_batched :: Property
prop_fc_batched = checkGradientNetB (fcNet @3 @3)

prop_quadcost_batched :: Property
prop_quadcost_batched = checkGradientNetB (quadCostNet @[2,2])

prop_sigmoid_batched :: Property
prop_sigmoid_batched = checkGradientNetB (sigmoidNet @[2,2])

prop_bias_batched :: Property
prop_bias_batched = checkGradientNetB (biasNet @[2,2])

prop_upper_triangular :: Property
prop_upper_triangular = checkGradientInvB (upperMultiplyInv @3)

prop_lower_triangular :: Property
prop_lower_triangular = checkGradientInvB (lowerMultiplyInv @3)

prop_splitchannel :: Property
prop_splitchannel = checkGradientInvB (splitChannelInv @2 @2 @2)

prop_affine :: Property
prop_affine = checkGradientInvB (affineCouplingInv @2 @1 @1 $ reshapeNet ~>> fcNet ~>> reshapeNet)

-- Make TemplateHaskell aware of above definitions.
$(return [])

main :: IO Bool
main = $(quickCheckAll)
