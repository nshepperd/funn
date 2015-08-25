{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}

import           Control.Applicative
import           Control.Monad
import           Data.Char
import           Data.Foldable
import           Data.IORef
import           Data.List
import           Data.Maybe
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           Data.Word
import           System.Environment

import           Options.Applicative

import           Text.Printf

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import qualified Data.Binary as LB
import qualified Data.ByteString.Lazy as LB

import           Control.DeepSeq
import           Data.Coerce
import           Debug.Trace
import           GHC.TypeLits
import           System.IO
import           System.IO.Unsafe

import           Data.Functor.Identity
import           Data.Random
import           Data.Random.Distribution.Categorical

import qualified Data.ByteString as B

import           Data.Vector (Vector)
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Unboxed as U
import qualified Numeric.LinearAlgebra.HMatrix as HM

import qualified Criterion

import           AI.Funn.Flat
import           AI.Funn.LSTM
import           AI.Funn.Mixing
import           AI.Funn.Network
import           AI.Funn.RNN
import           AI.Funn.SGD
import           AI.Funn.SomeNat
import           AI.Funn.Common

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

clipping_limit :: Double
clipping_limit = 0.1

descent' :: (VectorSpace s, s ~ (D s), Derivable s) => s -> Network Identity (s,i) s -> Parameters -> Network Identity (s,o) () -> Parameters -> IO (Vector i, Vector o) -> (Int -> s -> Parameters -> Parameters -> Double -> IO ()) -> IO ()
descent' initial_s layer p_layer_initial final p_final_initial source save = go initial_s p_layer_initial p_final_initial (0::Int) (Nothing, Nothing, Nothing)
  where
    go !s !p_layer !p_final !i (m_s, m_layer, m_final) = do
      (is, os) <- source
      let
        -- mid :: Network Identity s (Vector s)
        mid = feedR (V.init is) (rnnBig layer)
        -- loss :: Network Identity (Vector s, Vector c) ()
        loss = zipWithNetwork_ final
        lf = (-0.0002) :: Double
        ff = (-0.0002) :: Double
        Identity (ss, _,    kl) = evaluate mid p_layer s
        Identity ((), cost, kf) = evaluate loss p_final (ss,os)
        Identity ((dss, _), [dp_final']) = kf ()
        Identity (ds, [dp_layer']) = kl dss

        dp_gpn = (abs $ norm dp_layer' / norm p_layer)
        df_gpn = (abs $ norm dp_final' / norm p_final)

        -- Gradient clipping should avoid exploding gradients
        dp_layer = if dp_gpn > clipping_limit then
                     scaleParameters (clipping_limit / dp_gpn) dp_layer'
                   else dp_layer'

        dp_final = if df_gpn > clipping_limit then
                     scaleParameters (clipping_limit / df_gpn) dp_final'
                   else dp_final'

      when (i `mod` 100 == 0) $ do
        putStrLn $ "grad/param norm: " ++ show (dp_gpn, df_gpn)

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


sampleRNN :: Int -> s -> Network Identity (s, Blob 256) s -> Parameters -> Network Identity s (Blob 256) -> Parameters -> (s -> IO ()) -> IO String
sampleRNN n s layer p_layer final p_final xxx = go s n
  where
    go s 0 = return []
    go s n = do
      let Blob logps = runNetwork_ final p_final s :: Blob 256
          exps = V.map exp $ logps
          factor = 1 / V.sum exps
          ps = V.map (*factor) exps
      c <- sampleIO $ categorical [(p, chr i) | (i, p) <- zip [0..] (V.toList ps)]
      let new_s = runNetwork_ layer p_layer (s, oneofn V.! ord c)
      xxx new_s
      (c:) <$> unsafeInterleaveIO (go new_s (n-1))

    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

(>&>) :: (Monad m) => Network m (x1,a) (x2,b) -> Network m (y1,b) (y2,c) -> Network m ((x1,y1), a) ((x2,y2), c)
(>&>) one two = let two' = assocR >>> right two >>> assocL
                    one' = left swap >>> assocR >>> right one >>> assocL >>> left swap
                in one' >>> two'

data Options = Train (Maybe FilePath) FilePath FilePath
             | Sample FilePath (Maybe Int)
             | CheckDeriv
             deriving (Show)

type N = 1000
type Hidden = (Blob N, Blob N)

type LayerH h a b = Network Identity (h, a) (h, b)

instance LB.Binary (Blob n) where
  put (Blob xs) = putVector putDouble xs
  get = Blob <$> getVector getDouble

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  let optparser = (info (subparser $
                         command "train"
                         (info (Train
                                <$> optional (strOption (long "initial" <> action "file"))
                                <*> strOption (long "input" <> action "file")
                                <*> strOption (long "output" <> action "file"))
                          (progDesc "Train NN."))
                         <>
                         command "sample"
                         (info (Sample
                                <$> strOption (long "snapshot" <> action "file")
                                <*> optional (option auto (long "length")))
                          (progDesc "Sample output."))
                         <>
                         command "check"
                         (info (pure CheckDeriv)
                          (progDesc "Check Derivatives."))
                        )
                   fullDesc)

  opts <- customExecParser (prefs showHelpOnError) optparser

  let
    -- layer :: Network Identity ((Blob N, Blob N), Blob 256) (Blob N, Blob N)
    -- layer = assocR >>> right (mergeLayer >>> fcLayer >>> sigmoidLayer) >>> lstmLayer

    layer1 :: LayerH (Blob N) (Blob N, Blob 256) (Blob N)
    layer1 = right (mergeLayer >>> hierLayer >>> sigmoidLayer) >>> lstmLayer

    layer2 :: LayerH (Blob N) (Blob N) (Blob N)
    layer2 = right (hierLayer >>> sigmoidLayer) >>> lstmLayer

    layer :: Layer (((Blob N, Blob N), Blob N), Blob 256) ((Blob N, Blob N), Blob N)
    layer = assocR >>> (layer1 >&> layer2)

    finalx :: Network Identity (Hidden, Blob N) (Blob 256)
    finalx = left mergeLayer >>> mergeLayer >>> hierLayer

    final :: Network Identity ((Hidden, Blob N), Int) ()
    final = left finalx >>> softmaxCost

    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

    -- experiment :: Network Identity ((Blob N, Blob N), Vector (Blob 256)) (Blob 256)
    -- experiment = rnn layer >>> finalx

  print (params layer + params final)

  case opts of
   Train initpath input savefile -> do
     (initial, p_layer, p_final) <- case initpath of
       -- Just path -> read <$> readFile path
       Just path -> LB.decode <$> LB.readFile path
       Nothing -> (,,)
                  <$> pure ((unit, unit), unit)
                  <*> sampleIO (initialise layer)
                  <*> sampleIO (initialise final)

     deepseqM (initial, p_layer, p_final)

     text <- B.readFile input

     running_average <- newIORef 0
     running_count <- newIORef 0

     let
       α :: Double
       α = 1 - 1 / 500

       chunkSize = 50 :: Int

       tvec = V.fromList (B.unpack text) :: U.Vector Word8
       ovec = V.map (\c -> oneofn V.! (fromIntegral c)) (V.convert tvec) :: Vector (Blob 256)

       source :: IO (Vector (Blob 256), Vector Int)
       source = do s <- sampleIO (uniform 0 (V.length tvec - chunkSize))
                   let
                     input = V.slice s chunkSize ovec
                     output = V.map fromIntegral (V.convert (V.slice s chunkSize tvec))
                   return (input, output)

       save i init p_layer p_final c = do

         when (not (isInfinite c)) $ do
           modifyIORef' running_average (\x -> (α*x + (1 - α)*c))
           modifyIORef' running_count (\x -> (α*x + (1 - α)*1))

         x <- do q <- readIORef running_average
                 w <- readIORef running_count
                 return ((q / w) / fromIntegral (chunkSize - 1))

         let
           pa = V.sum (V.map abs $ getParameters p_layer) / fromIntegral (params layer)
           pa_max = V.maximum (V.map abs $ getParameters p_layer)
         when (pa > 1) $ do
           putStrLn $ printf "[%i] p_layer = %f" i pa

         let
           pf = V.sum (V.map abs $ getParameters p_final) / fromIntegral (params final)
           pf_max = V.maximum (V.map abs $ getParameters p_final)
         when (pf > 1) $ do
           putStrLn $ printf "[%i] p_final = %f" i pf

         when (i `mod` 50 == 0) $ do
           putStrLn $ show i ++ " " ++ show x ++ " " ++ show (c / fromIntegral (chunkSize-1)) ++ "    [" ++ show ((pa, pa_max), (pf, pf_max)) ++ "]"
         when (i `mod` 1000 == 0) $ do
           -- writeFile savefile $ show (init, p_layer, p_final)
           LB.writeFile (printf "%s-%6.6i-%5.5f.bin" savefile i x) $ LB.encode (init, p_layer, p_final)
           LB.writeFile (savefile ++ "-latest.bin") $ LB.encode (init, p_layer, p_final)
           test <- sampleRNN 100 init layer p_layer finalx p_final (\_ -> pure())
           putStrLn test

     deepseqM (tvec, ovec)

     descent' initial layer p_layer final p_final source save

   Sample initpath length -> do
     (initial, p_layer, p_final) <- LB.decode <$> LB.readFile initpath
     deepseqM (initial, p_layer, p_final)

     -- (\_ -> pure())
     text <- sampleRNN (fromMaybe maxBound length) initial layer p_layer finalx p_final (\_ -> pure())
     -- (\((x, y), z) -> print $ map (V.sum . V.map abs) [getBlob x,getBlob y,getBlob z])
     -- text <- sampleRNN (fromMaybe maxBound length) initial layer p_layer finalx p_final
     putStrLn text

   CheckDeriv -> do
    -- checkGradient $ splitLayer >>> rnn layer inputs >>> feedR o final

    -- checkGradient $ feedR o (softmaxCost :: Network Identity (Blob 256, Int) ())

    checkGradient $ splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

    checkGradient $ (fcLayer :: Network Identity (Blob 20) (Blob 20)) >>> splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

    checkGradient $ splitLayer >>> (lstmLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1)) >>> quadraticCost

    checkGradient $ splitLayer >>> left (freeLayer :: Layer (Blob 50) (Blob 100)) >>> (quadraticCost :: Network Identity (Blob 100, Blob 100) ())

    checkGradient $ splitLayer >>> left (hierLayer :: Layer (Blob 50) (Blob 100)) >>> (quadraticCost :: Network Identity (Blob 100, Blob 100) ())

    let benchNetwork net v = do
          pars <- sampleIO (initialise net)
          let f x = runIdentity $ do (o, c, k) <- evaluate net pars x
                                     (da, dp) <- k unit
                                     return (o, c, da, dp)
          -- return $ Criterion.nf (runNetwork_ net pars) v
          return $ Criterion.nf f v

    Criterion.benchmark =<< benchNetwork (freeLayer >>> biasLayer :: Network Identity (Blob 511) (Blob 511)) unit
    Criterion.benchmark =<< benchNetwork (fcLayer :: Network Identity (Blob 511) (Blob 511)) unit
    Criterion.benchmark =<< benchNetwork (hierLayer :: Network Identity (Blob 511) (Blob 511)) unit
