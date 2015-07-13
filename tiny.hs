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
import           Data.Word

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Control.DeepSeq
import           Debug.Trace
import           GHC.TypeLits
import           System.IO

import           Data.Functor.Identity
import           Data.Random

import qualified Data.ByteString as B

import           Data.Vector (Vector)
import           Data.Vector.Generic ((!))
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
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

runRNN :: (Monad m) => s -> Network m (s,i) s -> Parameters -> Network m (s,o) () -> Parameters -> [i] -> o -> m (Double, D s, D Parameters, D Parameters)
runRNN s layer p_layer final p_final [] o = do ((), cost, k) <- evaluate final p_final (s, o)
                                               ((ds, _), l_dp_final) <- k ()
                                               return (cost, ds, Parameters (V.replicate (params layer) 0), fold l_dp_final)

runRNN s layer p_layer final p_final (i:is) o = do (s_new, _, k) <- evaluate layer p_layer (s, i)
                                                   (cost, ds, dp_layer, dp_final) <- runRNN s_new layer p_layer final p_final is o
                                                   ((ds2, _), l_dp_layer2) <- k ds
                                                   let dp_layer2 = fold l_dp_layer2
                                                   return (cost, ds2, dp_layer `addParameters` dp_layer2, dp_final)

descent :: (VectorSpace s, s ~ (D s), Derivable s) => s -> Network Identity (s,i) s -> Parameters -> Network Identity (s,o) () -> Parameters -> IO ([i], o) -> IO ()
descent initial_s layer p_layer_initial final p_final_initial source = go initial_s p_layer_initial p_final_initial
  where
    go s p_layer p_final = do (is, o) <- source
                              let
                                n = fromIntegral (length is) :: Double
                                Identity (cost, ds, dp_layer, dp_final) = runRNN s layer p_layer final p_final is o
                              print cost
                              let
                                new_s = s ## scale (-0.01) ds
                                new_p_layer = p_layer `addParameters` scaleParameters (-0.01 / sqrt (1 + n)) dp_layer
                                new_p_final = p_final `addParameters` scaleParameters (-0.01) dp_final
                              go new_s new_p_layer new_p_final

checkGradient :: forall a. (KnownNat a) => Network Identity (Blob a) () -> IO ()
checkGradient network = do parameters <- sampleIO (initialise network)
                           input <- sampleIO (generateBlob $ uniform 0 1)
                           let (e, d_input, d_parameters) = runNetwork' network parameters input
                           d1 <- sampleIO (V.replicateM                a (uniform (-ε) ε))
                           d2 <- sampleIO (V.replicateM (params network) (uniform (-ε) ε))
                           let parameters' = Parameters (V.zipWith (+) (getParameters parameters) d2)
                               input' = input ## Blob d1
                           let (e', _, _) = runNetwork' network parameters' input'
                               δ_expected = sum (V.toList $ V.zipWith (*) (getBlob d_input) d1)
                                            + sum (V.toList $ V.zipWith (*) (getParameters d_parameters) d2)
                           print (e' - e, δ_expected)

  where
    a = fromIntegral (natVal (Proxy :: Proxy a)) :: Int
    ε = 0.000001

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  -- withNat 2 (\(Proxy :: Proxy n) -> do
  --               let network :: Network Identity (Blob n) (Blob 1)
  --                   network = fcLayer >>> (fcLayer :: Network Identity (Blob 3) (Blob 1))

  --               params <- sampleIO (initialise network)
  --               print params
  --               print $ runNetwork network params (blob [1, 2]))

  let
    layer :: Network Identity ((Blob 100, Blob 100), Blob 256) (Blob 100, Blob 100)
    layer = assocR >>> right (mergeLayer >>> fcLayer >>> sigmoidLayer) >>> lstmLayer
            -- (left fcLayer >>> right fcLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1))
            -- (lstmLayer :: Network Identity (Blob 100, Blob 40) (Blob 100, Blob 100))

    -- layer = layerWithSize (Proxy :: Proxy 100)

    final :: Network Identity ((Blob 100, Blob 100), Int) ()
    final = left (mergeLayer >>> (fcLayer :: Network Identity (Blob 200) (Blob 256))) >>> softmaxCost

  p_layer <- sampleIO (initialise layer)
  p_final <- sampleIO (initialise final)
  -- print p_layer
  -- print p_final

  text <- B.readFile "tiny.hs"

  let
    initial = (unit, unit)

    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

    tvec = V.fromList (B.unpack text) :: U.Vector Word8
    ovec = V.map (\c -> oneofn V.! (fromIntegral c)) (V.convert tvec) :: Vector (Blob 256)
    source :: IO ([Blob 256], Int)
    source = do s <- sampleIO (uniform 0 (V.length tvec - 20))
                l <- sampleIO (uniform 1 20)
                let
                  input = V.toList $ V.slice s (l-1) ovec
                  output = fromIntegral (tvec V.! (s + l - 1))
                return (input, output)

  descent initial layer p_layer final p_final source

  -- checkGradient $ splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

  -- checkGradient $ splitLayer >>> (lstmLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1)) >>> quadraticCost
