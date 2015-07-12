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

addParameters :: Parameters -> Parameters -> Parameters
addParameters (Parameters x) (Parameters y) = Parameters (x + y)

runRNN :: (VectorSpace ds, ds ~ (D s), Derivable s, Monad m) => s -> Network m (s,i) s -> Parameters -> Network m (s,o) () -> Parameters -> [i] -> o -> m (Double, D s, D Parameters, D Parameters)
runRNN s layer p_layer final p_final [] o = do ((), cost, k) <- evaluate final p_final (s, o)
                                               ((ds, _), l_dp_final) <- k ()
                                               return (cost, ds, Parameters (V.replicate (params layer) 0), fold l_dp_final)

runRNN s layer p_layer final p_final (i:is) o = do (s_new, _, k) <- evaluate layer p_layer (s, i)
                                                   (cost, ds, dp_layer, dp_final) <- runRNN s_new layer p_layer final p_final is o
                                                   ((ds2, _), l_dp_layer2) <- k ds
                                                   let dp_layer2 = fold l_dp_layer2
                                                   return (cost, ds ## ds2, addParameters dp_layer dp_layer2, dp_final)


main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  withNat 2 (\(Proxy :: Proxy n) -> do
                let network :: Network Identity (Blob n) (Blob 1)
                    network = fcLayer >>> (fcLayer :: Network Identity (Blob 3) (Blob 1))

                params <- sampleIO (initialise network)
                print params
                print $ runNetwork network params (blob [1, 2]))

  let layer :: Network Identity ((Blob 1, Blob 1), Blob 2) (Blob 1, Blob 1)
      layer = assocR >>> right (mergeLayer >>> fcLayer >>> sigmoidLayer) >>>
              (lstmLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1))

      final :: Network Identity ((Blob 1, Blob 1), Blob 1) ()
      final = left (mergeLayer >>> fcLayer >>> sigmoidLayer) >>> quadraticCost

  p_layer <- sampleIO (initialise layer)
  p_final <- sampleIO (initialise final)
  print p_layer
  print p_final


  let
    initial :: (Blob 1, Blob 1)
    initial = (runIdentity $ generateBlob (pure 0),
               runIdentity $ generateBlob (pure 0))

  print $ runRNN initial layer p_layer final p_final [blob [0, 1]] (blob [0.4])
  -- print $ runNetwork network params (blob [1], blob [0.3, 0.3, 0.2, 0.5])
  -- print $ runNetwork' network' params (blob [1], blob [0.3, 0.3, 0.2, 0.5])
