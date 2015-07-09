{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import           Control.Applicative
import           Data.Foldable
import           Data.Traversable
import           Data.List
import           Data.Monoid
import           Data.Proxy

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Control.DeepSeq
import           GHC.TypeLits
import           System.IO

import           Data.Functor.Identity
import           Data.Random

import           Data.Vector (Vector)
import           Data.Vector.Generic ((!))
import qualified Data.Vector.Generic as V
import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Flat
import           AI.Funn.Network
import           AI.Funn.SomeNat

type Layer = Network Identity

sampleIO :: RVar a -> IO a
sampleIO v = runRVar v StdRandom

blob :: [Double] -> Blob n
blob xs = Blob (V.fromList xs)

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  let network :: forall n. (KnownNat n) => Network Identity (Blob n) (Blob 1)
      network = fcLayer >>> (fcLayer :: Network Identity (Blob 3) (Blob 1))
      network1 :: Network Identity (Blob 2) (Blob 1)
      network1 = network
      network2 :: Network Identity SBlob (Blob 1)
      network2 = weakenL 2 network

  params <- sampleIO (initialise network1)
  print params
  print $ runNetwork network1 params (blob [1, 2])

  params2 <- sampleIO (initialise network2)
  print params2
  print $ runNetwork network2 params2 (SBlob (V.fromList [1, 2]))
