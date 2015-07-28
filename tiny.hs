{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

import           Control.Applicative
import           Control.Monad
import           Data.Char
import           Data.Foldable
import           Data.IORef
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           Data.Word
import           System.Environment

import           Options.Applicative

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Control.DeepSeq
import           Data.Coerce
import           Debug.Trace
import           GHC.TypeLits
import           System.IO

import           Data.Functor.Identity
import           Data.Random
import           Data.Random.Distribution.Categorical

import qualified Data.ByteString as B

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import qualified Numeric.LinearAlgebra.HMatrix as HM

import           AI.Funn.Flat
import           AI.Funn.LSTM
import           AI.Funn.Network
import           AI.Funn.RNN
import           AI.Funn.SomeNat

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

norm :: Parameters -> Double
norm (Parameters xs) = sqrt $ V.sum $ V.map (^2) xs

descent :: (VectorSpace s, s ~ (D s), Derivable s) => s -> Network Identity (s,i) s -> Parameters -> Network Identity (s,o) () -> Parameters -> IO ([i], o) -> (Int -> s -> Parameters -> Parameters -> Double -> IO ()) -> IO ()
descent initial_s layer p_layer_initial final p_final_initial source save = go initial_s p_layer_initial p_final_initial 0 (Nothing, Nothing, Nothing)
  where
    go !s !p_layer !p_final !i (m_s, m_layer, m_final) = do
      (is, o) <- source
      let
        n = fromIntegral (length is) :: Double
        lf = (-0.01) :: Double
        ff = (-0.01) :: Double
      (cost, ds, dp_layer, dp_final) <- runRNNIO s layer p_layer final p_final is o
      -- let Identity (cost, ds, dp_layer, dp_final) = runRNN' s layer p_layer final p_final is o
      let
        gpn = abs $ norm dp_layer / norm p_layer
      when (i `mod` 100 == 0) $ do
        putStrLn $ "grad/param norm: " ++ show gpn
      save i s p_layer p_final cost
      let
        -- (new_s, new_m_s) = let δ = case m_s of
        --                             Just m -> scale (-0.01) ds ## scale (0.9) m
        --                             Nothing -> scale (-0.01) ds
        --                    in (s ## δ, Just δ)
        (new_p_layer, new_m_layer) = momentum lf p_layer dp_layer m_layer
        (new_p_final, new_m_final) = momentum ff p_final dp_final m_final
      go s new_p_layer new_p_final (i+1) (m_s, new_m_layer, new_m_final)

    momentum :: Double -> Parameters -> Parameters -> Maybe Parameters -> (Parameters, Maybe Parameters)
    momentum f par d_par m_par = let δ = case m_par of
                                          Just m -> scaleParameters f d_par `addParameters` scaleParameters 0.9 m
                                          Nothing -> scaleParameters f d_par
                                 in (par `addParameters` δ, Just δ)

feedR :: (Monad m) => b -> Network m (a,b) c -> Network m a c
feedR b network = Network ev (params network) (initialise network)
  where
    ev pars a = do (c, cost, k) <- evaluate network pars (a, b)
                   let backward dc = do
                         ((da, _), dpar) <- k dc
                         return (da, dpar)
                   return (c, cost, backward)

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


sampleRNN :: Int -> s -> Network Identity (s, Blob 256) s -> Parameters -> Network Identity s (Blob 256) -> Parameters -> IO String
sampleRNN n s layer p_layer final p_final = go s n
  where
    go s 0 = return []
    go s n = do
      let Blob logps = runNetwork_ final p_final s :: Blob 256
          exps = V.map exp $ logps
          factor = 1 / V.sum exps
          ps = V.map (*factor) exps
      c <- sampleIO $ categorical [(p, chr i) | (i, p) <- zip [0..] (V.toList ps)]
      let new_s = runNetwork_ layer p_layer (s, oneofn V.! ord c)
      (c:) <$> go new_s (n-1)

    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

(>&>) :: (Monad m) => Network m (x1,a) (x2,b) -> Network m (y1,b) (y2,c) -> Network m ((x1,y1), a) ((x2,y2), c)
(>&>) one two = let two' = assocR >>> right two >>> assocL
                    one' = left swap >>> assocR >>> right one >>> assocL >>> left swap
                in one' >>> two'

data Options = Options {
  initial :: Maybe FilePath,
  input :: FilePath,
  output :: FilePath
  } deriving (Show)

type N = 10
type Hidden = (Blob N, Blob N)

type LayerH h a b = Network Identity (h, a) (h, b)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  opts <- execParser (info (Options
                            <$> optional (strOption (long "initial"))
                            <*> strOption (long "input")
                            <*> strOption (long "output"))
                      fullDesc)

  let
    -- layer :: Network Identity ((Blob N, Blob N), Blob 256) (Blob N, Blob N)
    -- layer = assocR >>> right (mergeLayer >>> fcLayer >>> sigmoidLayer) >>> lstmLayer

    layer1 :: LayerH (Blob N) (Blob N, Blob 256) (Blob N)
    layer1 = right (mergeLayer >>> fcLayer >>> sigmoidLayer) >>> lstmLayer

    layer2 :: LayerH (Blob N) (Blob N) (Blob N)
    layer2 = right (fcLayer >>> sigmoidLayer) >>> lstmLayer

    layer :: Layer (((Blob N, Blob N), Blob N), Blob 256) ((Blob N, Blob N), Blob N)
    layer = assocR >>> (layer1 >&> layer2)

    finalx :: Network Identity (Hidden, Blob N) (Blob 256)
    finalx = left mergeLayer >>> mergeLayer >>> fcLayer

    final :: Network Identity ((Hidden, Blob N), Int) ()
    final = left finalx >>> softmaxCost

    -- experiment :: Network Identity ((Blob N, Blob N), Vector (Blob 256)) (Blob 256)
    -- experiment = rnn layer >>> finalx

  let fname = input opts
      savefile = output opts

  (initial, p_layer, p_final) <- case initial opts of
    Just path -> read <$> readFile path
    Nothing -> (,,)
               <$> pure ((unit, unit), unit)
               <*> sampleIO (initialise layer)
               <*> sampleIO (initialise final)

  text <- B.readFile fname

  let α = 0.98
  running_average <- newIORef 0

  let
    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

    tvec = V.fromList (B.unpack text) :: U.Vector Word8
    ovec = V.map (\c -> oneofn V.! (fromIntegral c)) (V.convert tvec) :: Vector (Blob 256)
    source :: IO ([Blob 256], Int)
    source = do s <- sampleIO (uniform 0 (V.length tvec - 20))
                l <- sampleIO (uniform 1 19)
                let
                  input = V.toList $ V.slice s l ovec
                  output = fromIntegral (tvec V.! (s + l))
                return (input, output)

    save i init p_layer p_final c = do
      modifyIORef' running_average (\x -> (α*x + (1 - α)*c))
      when (i `mod` 50 == 0) $ do
        x <- readIORef running_average
        putStrLn $ show i ++ " " ++ show x ++ " " ++ show c
      when (i `mod` 1000 == 0) $ do
        writeFile savefile $ show (init, p_layer, p_final)
        test <- sampleRNN 100 init layer p_layer finalx p_final
        putStrLn test

  deepseqM (tvec, ovec)

  descent initial layer p_layer final p_final source save



  -- (inputs, o) <- source
  -- checkGradient $ splitLayer >>> rnn layer inputs >>> feedR o final

  -- checkGradient $ feedR o (softmaxCost :: Network Identity (Blob 256, Int) ())

  -- checkGradient $ splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

  -- checkGradient $ (fcLayer :: Network Identity (Blob 20) (Blob 20)) >>> splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

  -- checkGradient $ splitLayer >>> (lstmLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1)) >>> quadraticCost

  -- checkGradient $ splitLayer >>> left splitLayer >>> layer >>> quadraticCost
