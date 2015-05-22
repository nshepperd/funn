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

type Layer = Network Identity

sampleIO :: RVar a -> IO a
sampleIO v = runRVar v StdRandom

blob :: [Double] -> Blob n
blob xs = Blob (V.fromList xs)

totalError :: Network Identity a () -> Parameters -> Vector a -> (Double, Parameters)
totalError network params trainSet = let (errors, _, gparams) = unzip3 $ map (runNetwork' network params) (V.toList trainSet)
                                         dparam = foldl' (\a (Parameters b) -> V.zipWith (+) a b) (V.replicate d 0) gparams
                                         total = sum errors
                                     in
                                      (total, Parameters dparam)
  where
    d = V.length (getParameters params)

gradientDescent :: Network Identity a () -> Parameters -> Vector a -> Parameters
gradientDescent network params trainSet = let (_, Parameters δs) = totalError network params trainSet
                                              xs = getParameters params
                                          in
                                           Parameters (V.zipWith (\x δ -> x - ε * δ / fromIntegral n) xs δs)
  where
    n = V.length trainSet
    ε = 0.01

type DV = 4
dv = 4

type ContextSize = 1
contextSize = 1

sumBlobs :: (KnownNat n) => [Blob n] -> Blob n
sumBlobs = foldl' (\(Blob a) (Blob b) -> Blob (V.zipWith (+) a b)) unit

shiftTag :: Blob n -> Blob n -> Blob n
shiftTag (Blob xs) (Blob δs) = Blob (V.zipWith (\x δ -> x - ε * δ) xs δs)
  where
    ε = 0.01

shiftParams :: Parameters -> Parameters -> Parameters
shiftParams (Parameters xs) (Parameters δs) = Parameters (V.zipWith (\x δ -> x - ε * δ) xs δs)
  where
    ε = 0.01

npairs :: Int -> [a] -> [([a], a)]
npairs n xs = go xs (drop n xs)
  where go _ [] = []
        go cs (x:xs) = (take n cs, x) : go (tail cs) xs

deepseqM :: (Monad m, NFData a) => a -> m ()
deepseqM x = deepseq x (return ())

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  let network :: Layer (Blob (ContextSize * DV)) (Blob 4)
      network = fcLayer
      training = left network >>> softmaxCost

      trainText = "abcdabcddcba"
      dictionary = V.fromList "abcd" :: Vector Char
      idict = Map.fromList (zip (V.toList dictionary) [1..]) :: Map Char Int

      trainSet :: Vector ([Int], Int)
      trainSet = V.fromList $ npairs contextSize $ replicate contextSize 0 ++ map (idict Map.!) trainText

  let
    go :: Parameters -> Vector (Blob DV) -> IO ()
    go params tags = do
      i <- sampleIO (uniform 0 (V.length trainSet - 1))
      let (cs, target) = trainSet!i
          context = Blob $ V.concat [getBlob (tags!j) | j <- cs]

      let (cost, (dtags, ()), dpars) = runNetwork' training params (context, target-1)
          dtags' = [Blob $ V.slice (i * dv) dv (getBlob dtags) | i <- [0 .. contextSize - 1]] :: [Blob DV]
          dtags2 = [(j, sumBlobs [δ | (i, δ) <- zip cs dtags', i == j]) | j <- nub cs] :: [(Int, Blob DV)]
          tags' = tags V.// [(j, shiftTag (tags!j) δ) | (j, δ) <- dtags2]
          params' = shiftParams params dpars

      print cost
      deepseqM (params', tags')

      go params' tags'

  params <- sampleIO (initialise network)
  tags <- sampleIO (V.replicateM 5 (generateBlob (uniform (-1) 1))) :: IO (Vector (Blob DV))
  go params tags
