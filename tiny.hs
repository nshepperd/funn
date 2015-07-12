{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import           Control.Applicative
import           Control.Monad
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

scaleParameters :: Double -> Parameters -> Parameters
scaleParameters x (Parameters y) = Parameters (HM.scale x y)

runRNN :: (VectorSpace ds, ds ~ (D s), Derivable s, Monad m) => s -> Network m (s,i) s -> Parameters -> Network m (s,o) () -> Parameters -> [i] -> o -> m (Double, D s, D Parameters, D Parameters)
runRNN s layer p_layer final p_final [] o = do ((), cost, k) <- evaluate final p_final (s, o)
                                               ((ds, _), l_dp_final) <- k ()
                                               return (cost, ds, Parameters (V.replicate (params layer) 0), fold l_dp_final)

runRNN s layer p_layer final p_final (i:is) o = do (s_new, _, k) <- evaluate layer p_layer (s, i)
                                                   (cost, ds, dp_layer, dp_final) <- runRNN s_new layer p_layer final p_final is o
                                                   ((ds2, _), l_dp_layer2) <- k ds
                                                   let dp_layer2 = fold l_dp_layer2
                                                   return (cost, ds ## ds2, addParameters dp_layer dp_layer2, dp_final)

descent :: (VectorSpace s, s ~ (D s), Derivable s) => s -> Network Identity (s,i) s -> Parameters -> Network Identity (s,o) () -> Parameters -> IO ([i], o) -> IO ()
descent initial_s layer p_layer_initial final p_final_initial source = go initial_s p_layer_initial p_final_initial
  where
    go s p_layer p_final = do (is, o) <- source
                              let
                                n = fromIntegral (length is)
                                Identity (cost, ds, dp_layer, dp_final) = runRNN s layer p_layer final p_final is o
                              print cost
                              let
                                new_s = s ## scale (-0.01) ds
                                new_p_layer = p_layer `addParameters` scaleParameters (-0.1 / sqrt (1 + n)) dp_layer
                                new_p_final = p_final `addParameters` scaleParameters (-0.01) dp_final
                              go new_s new_p_layer new_p_final


main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  -- withNat 2 (\(Proxy :: Proxy n) -> do
  --               let network :: Network Identity (Blob n) (Blob 1)
  --                   network = fcLayer >>> (fcLayer :: Network Identity (Blob 3) (Blob 1))

  --               params <- sampleIO (initialise network)
  --               print params
  --               print $ runNetwork network params (blob [1, 2]))

  let layer :: Network Identity ((Blob 1, Blob 1), Blob 2) (Blob 1, Blob 1)
      layer = assocR >>> right (mergeLayer >>> fcLayer >>> sigmoidLayer) >>>
              (lstmLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1))

      final :: Network Identity ((Blob 1, Blob 1), Blob 1) ()
      final = left (mergeLayer >>> fcLayer >>> sigmoidLayer) >>> quadraticCost

  p_layer <- sampleIO (initialise layer)
  p_final <- sampleIO (initialise final)
  -- print p_layer
  -- print p_final


  let
    initial :: (Blob 1, Blob 1)
    initial = (blob [0], blob [0])

    source' :: IO ([Blob 2], Blob 1)
    source' = do n <- sampleIO (uniform 1 3) :: IO Int
                 pieces <- replicateM n (generateBlob $ sampleIO $ uniform 0 1) :: IO [Blob 2]
                 let
                   result = getBlob (pieces !! 0) V.! 0
                 -- let result = sum (map (V.sum . getBlob) pieces) / (2 * fromIntegral n)
                 return (pieces, blob [result])

  samples <- V.replicateM 10 source' :: IO (Vector ([Blob 2], Blob 1))

  let
    source = do n <- sampleIO (uniform 0 (V.length samples - 1))
                return (samples V.! n)

  descent initial layer p_layer final p_final source
  -- print $ runRNN initial layer p_layer final p_final [blob [0, 1]] (blob [0.4])

  -- print $ runNetwork network params (blob [1], blob [0.3, 0.3, 0.2, 0.5])
  -- print $ runNetwork' network' params (blob [1], blob [0.3, 0.3, 0.2, 0.5])
