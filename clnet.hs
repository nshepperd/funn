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
import           Control.Concurrent
import           Control.DeepSeq
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Char
import           Data.Foldable
import           Data.List
import           Data.Maybe
import           Data.Monoid
import           Data.Proxy
import           Data.Random
import           Data.Random.List
import           Data.Traversable
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

import           AI.Funn.CL.Batched.Layers.FullyConnected
import           AI.Funn.CL.Batched.Layers.Simple
import           AI.Funn.CL.Batched.Network
import           AI.Funn.CL.Blob (Blob(..))
import qualified AI.Funn.CL.Blob as Blob
import qualified AI.Funn.CL.LazyMem as LM
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Common
import           AI.Funn.Diff.Diff (Diff(..), Derivable(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as F
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Optimizer.Adam
import           AI.Funn.Space
import           AI.Funn.TypeLits

-- Batch Size.
type W = 10

fixSize :: forall a c. Indexed c => c 0 (Tensor '[W, a]) (Tensor '[W, a])
fixSize = iid

mainNetwork :: Network IO W _ (Tensor '[W, 2]) (Tensor '[W, 1])
mainNetwork = fcNet ~>> sigmoidNet ~>> (fixSize @3) ~>> fcNet

evalNetwork :: Network IO W _ (Tensor '[W, 2], Tensor '[W, 1]) (Tensor '[W])
evalNetwork = first mainNetwork ~>> quadCostNet

catMany :: [Tensor ds] -> Tensor (ω ': ds)
catMany tensors = unsafePerformIO (Tensor <$> LM.toStrict buffers)
  where
    buffers = foldMap (LM.fromStrict . getTensor) tensors

collect :: IO (Tensor '[2], Tensor '[1]) -> IO (Tensor '[W, 2], Tensor '[W, 1])
collect dataset = do
  items <- replicateM 10 dataset
  return (catMany (map fst items), catMany (map snd items))

train :: Tensor _ -> IO (Tensor '[2], Tensor '[1]) -> IO ()
train initialPar dataset = do
  ones <- Tensor.replicate 1
  let
    objective par = do
      pair <- collect dataset
      (o, k) <- runNetwork evalNetwork (par, pair)
      (dpar, _) <- k ones
      oo <- sum <$> Tensor.toList o
      return (oo / 10, averageGrad dpar)

    go av trainState = do
      (o, grad) <- objective (extractAdam trainState)
      trainState' <- updateAdam grad trainState
      print av
      go (updateRunningAverage o av) trainState'

  trainState <- initAdam 0.01 0.9 0.999 1e-8 initialPar :: IO (AdamState IO (Tensor _) (Tensor _))
  go (newRunningAverage 0.99) trainState

fromFlat :: (MonadIO m, KnownNat n) => F.Blob n -> m (Tensor '[n])
fromFlat blob = Tensor.fromList (F.toList blob)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering
  initOpenCL

  initial_par_blob <- sample (netInit mainNetwork)
  initial_par <- fromFlat initial_par_blob

  pairs <- traverse (\(a,b) -> (,) <$> Tensor.fromList a <*> Tensor.fromList b) [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])] :: IO [(Tensor '[2], Tensor '[1])]

  train initial_par (sample (randomElement pairs))
