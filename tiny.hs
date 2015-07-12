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
import           AI.Funn.LSTM

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

  withNat 2 (\(Proxy :: Proxy n) -> do
                let network :: Network Identity (Blob n) (Blob 1)
                    network = fcLayer >>> (fcLayer :: Network Identity (Blob 3) (Blob 1))

                params <- sampleIO (initialise network)
                print params
                print $ runNetwork network params (blob [1, 2]))

  let network :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1)
      network = lstmLayer
      network' = network >>> quadraticCost

  params <- sampleIO (initialise network)
  print params
  print $ runNetwork network params (blob [1], blob [0.3, 0.3, 0.2, 0.5])
  print $ runNetwork' network' params (blob [1], blob [0.3, 0.3, 0.2, 0.5])
