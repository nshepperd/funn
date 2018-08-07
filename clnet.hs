{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE TypeApplications, PartialTypeSignatures #-}
{-# LANGUAGE FlexibleInstances, MultiParamTypeClasses #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE ConstraintKinds #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module Main where

import           Control.Applicative
import           Control.Category
import           Control.Concurrent
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.IO.Class
import qualified Control.Monad.State.Lazy as SL
import           Control.Monad.Trans
import qualified Data.Binary as LB
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as LB
import qualified Data.ByteString.Lazy.Char8 as LC
import           Data.Char
import           Data.Foldable
import           Data.Functor.Identity
import           Data.IORef
import           Data.List
import           Data.Maybe
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Random.Distribution.Categorical
import           Data.Random.List
import           Data.Traversable
import           Data.Type.Equality ((:~:)(..))
import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import           Data.Word
import           Debug.Trace
import           GHC.TypeLits
import           Options.Applicative
import           System.Clock
import           System.Environment
import           System.IO
import           System.IO.Unsafe
import           System.Random
import           Text.Printf

import           AI.Funn.CL.Blob (Blob(..))
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.Layers.FullyConnected
import           AI.Funn.CL.Layers.Tensor
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Network
import           AI.Funn.CL.Tensor (Tensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import           AI.Funn.Diff.Pointed
import           AI.Funn.Diff.RNN
import qualified AI.Funn.Flat.Blob as F
import           AI.Funn.Flat.ParBox
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space
import           AI.Funn.TypeLits

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

fixSize :: forall a c. Indexed c => c 0 (Tensor '[a]) (Tensor '[a])
fixSize = iid

mainNetwork :: Network IO _ (Tensor '[2]) (Tensor '[1])
mainNetwork = fcNet ~>> sigmoidNet ~>> (fixSize @3) ~>> fcNet

evalNetwork :: Network IO _ (Tensor '[2], Tensor '[1]) Double
evalNetwork = first mainNetwork ~>> quadCostNet

train :: Tensor _ -> IO ()
train initialPar = do
    pairs <- traverse (\(a,b) -> (,) <$> Tensor.fromList a <*> Tensor.fromList b) [
      ([0, 0], [0]),
      ([0, 1], [1]),
      ([1, 0], [1]),
      ([1, 1], [0])] :: IO [(Tensor '[2], Tensor '[1])]
    let
      evalDiff = toDiff evalNetwork
      objective par = do
        pair <- sample (randomElement pairs)
        (o, k) <- runDiff evalDiff (par, pair)
        (dpar, _) <- k 1
        return (o, dpar)

      go av trainState = do
          (o, grad) <- objective (extractAdam trainState)
          trainState' <- updateAdam grad trainState
          print av
          go (updateRunningAverage o av) trainState'

    trainState <- initAdam 0.01 0.9 0.999 1e-8 initialPar :: IO (AdamState IO (Tensor _) (Tensor _))
    go (newRunningAverage 0.99) trainState

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  initOpenCL

  initial_par_blob <- sample (netInit mainNetwork)
  initial_par <- Tensor.fromList (F.toList initial_par_blob) :: IO (Tensor _)
  train initial_par
