{-# LANGUAGE KindSignatures, DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
module AI.Funn.Flat.Softmix (softmixDiff) where

import           Control.Applicative
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable
import qualified Data.Vector.Generic as V
import           GHC.Float
import           GHC.TypeLits

import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Additive(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Flat.Blob (Blob(..), blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Space

softmixDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob n, Blob n) Double
softmixDiff = Diff run
  where run (bxs, bys)
          = let xs = getBlob bxs
                ys = getBlob bys
                exp_total = V.sum (V.map exp xs)
                ps = V.map (\x -> exp x / exp_total) xs
                po = V.sum (V.zipWith (*) ps ys)
                cost = -log po
            in return (cost, backward ps ys po)

        backward ps ys po dcost =
          let
            back = V.zipWith (\pi yi -> pi * (1 - yi/po) * dcost) ps ys
            backy = V.map (\pi -> -pi / po * dcost) ps
          in return (blob back, blob backy)
