{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables, FlexibleContexts #-}
{-# LANGUAGE BangPatterns #-}

import           Control.Applicative
import           Control.Monad
import           Data.Foldable
import           Data.Monoid
import           Data.Traversable

import           Data.Char
import           Data.IORef
import           Data.List
import           Data.Maybe
import           Data.Proxy
import           Data.Word

import           System.Clock
import           System.Environment
import           System.IO

import           Options.Applicative

import           Text.Printf

import qualified Data.Binary as LB
import qualified Data.ByteString as B
import qualified Data.ByteString.Lazy as LB

import           Control.DeepSeq
import           Data.Coerce
import           Debug.Trace
import           GHC.TypeLits
import           System.IO.Unsafe

import qualified Control.Monad.State.Lazy as SL
import           Data.Functor.Identity
import           Data.Random
import           Data.Random.Distribution.Categorical
import           System.Random

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
clipping_limit = 0.02

clip :: Parameters -> Parameters -> Parameters
clip ps ds
  | V.any (\x -> isInfinite x || isNaN x) (getParameters ds) =
      trace ("Infinity in gradient, reducing") $ scaleParameters (0.01) ps
  | total > rel * clipping_limit =
      trace ("clipping " ++ show (total, rel*clipping_limit)) $ scaleParameters (clipping_limit*rel/total) ds
  | otherwise = ds
  where
    rel = sqrt $ fromIntegral (V.length (getParameters ds))
    -- xs1 = V.map (max (-50) . min 50) xs
    total = norm ds

dropL :: (da ~ D a, VectorSpace da) => Layer (a, b) b
dropL = Network ev 0 (pure mempty)
  where
    ev _ (a, b) = let backward db = return ((unit, db), [])
                      in return (b, 0, backward)

dropR :: (da ~ D a, VectorSpace da) => Layer (b, a) b
dropR = swap >>> dropL


dup :: (da ~ D a, VectorSpace da) => Layer a (a, a)
dup = Network ev 0 (pure mempty)
  where
    ev _ a = let backward (da1,da2) = return ((da1 ## da2), [])
             in return ((a,a), 0, backward)

descent :: (VectorSpace s, s ~ (D s), Derivable s) => s -> Network Identity (s,i) (s,t) -> Parameters -> Network Identity (t,o) () -> Parameters -> IO (Vector i, Vector o) -> (Int -> s -> Parameters -> Parameters -> Double -> IO ()) -> IO ()
descent initial_s layer p_layer_initial final p_final_initial source save = go initial_s p_layer_initial p_final_initial (0::Int) (Nothing, Nothing, Nothing)
  where
    go !s !p_layer !p_final !i (m_s, m_layer, m_final) = do
      (is, os) <- source
      let
        -- mid :: Network Identity s (Vector t)
        mid = feedR is (rnnX layer) >>> dropL
        -- loss :: Network Identity (Vector t, Vector o) ()
        loss = zipWithNetwork_ final
        lf = (-0.01) :: Double
        ff = (-0.01) :: Double
        Identity (ts, _,    kl) = evaluate mid p_layer s
        Identity ((), cost, kf) = evaluate loss p_final (ts,os)
        Identity ((dts, _), [dp_final']) = kf ()
        Identity (ds, [dp_layer']) = kl dts

        dp_gpn = (abs $ norm dp_layer / norm p_layer)
        df_gpn = (abs $ norm dp_final / norm p_final)

        -- Gradient clipping should avoid exploding gradients
        dp_layer = clip p_layer dp_layer'
        dp_final = clip p_final dp_final'

      when (i `mod` 100 == 0) $ do
        putStrLn $ "grad/param norm: " ++ show (dp_gpn, df_gpn)
        putStrLn $ "grad norm: " ++ show (norm dp_layer', norm dp_final', params layer, params final)

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


sampleRNN :: s -> (Blob 256) -> Network Identity (s, Blob 256) (s, t) -> Parameters -> Network Identity t (Blob 256) -> Parameters -> [Word8]
sampleRNN s cfirst layer p_layer final p_final = SL.evalState (go s cfirst) (mkStdGen 1)
  where
    go s cprev = do
      let (new_s, t) = runNetwork_ layer p_layer (s, cprev)
          logps = getBlob $ runNetwork_ final p_final t
          exps = V.map exp $ logps
          factor = 1 / V.sum exps
          ps = V.map (*factor) exps
      c <- runRVar (categorical $ zip (V.toList ps) [0..]) StdRandom
      rest <- go new_s (oneofn V.! fromIntegral c)
      return (c : rest)

    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

(>&>) :: (Monad m) => Network m (x1,a) (x2,b) -> Network m (y1,b) (y2,c) -> Network m ((x1,y1), a) ((x2,y2), c)
(>&>) one two = let two' = assocR >>> right two >>> assocL
                    one' = left swap >>> assocR >>> right one >>> assocL >>> left swap
                in one' >>> two'

data LayerChoice = FCLayer | HierLayer Int
                 deriving (Show)

data Options = Options LayerChoice Commands
             deriving (Show)

data Commands = Train (Maybe FilePath) FilePath (Maybe FilePath) (Maybe FilePath)
              | Sample FilePath (Maybe Int)
              | CheckDeriv
              deriving (Show)

type N = 256
type LayerH h a b = Network Identity (h, a) (h, b)

instance LB.Binary (Blob n) where
  put (Blob xs) = putVector putDouble xs
  get = Blob <$> getVector getDouble

stack :: (Monad m) => Network m (s, a) (s, b) -> Network m (t, b) (t, c) -> Network m ((s,t), a) ((s,t), c)
stack one two = left swap >>> assocR -- (t, (s,a))
                >>> right one -- (t, (s,b))
                >>> assocL >>> left swap >>> assocR -- (s, (t, b))
                >>> right two -- (s, (t, c))
                >>> assocL -- ((s,t), c)

type H = (Blob N, Blob N)

main :: IO ()
main = do
  hSetBuffering stdout LineBuffering

  let optparser = (info (Options
                         <$> (const FCLayer <$> switch (long "fclayer")
                              <|> HierLayer <$> option auto (long "hierlayer"))
                         <*> (subparser $
                              command "train"
                              (info (Train
                                     <$> optional (strOption (long "initial" <> action "file"))
                                     <*> strOption (long "input" <> action "file")
                                     <*> optional (strOption (long "output" <> action "file"))
                                     <*> optional (strOption (long "log" <> action "file"))
                                    )
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
                             ))
                         fullDesc)

  opts <- customExecParser (prefs showHelpOnError) optparser

  let
    connectingLayer :: (KnownNat a, KnownNat b, Monad m) => Network m (Blob a) (Blob b)
    connectingLayer = case opts of
                       Options FCLayer _ -> fcLayer
                       Options (HierLayer n) _ -> hierLayerN n

    layer1 :: Layer ((Blob N, Blob N), Blob 256) (Blob N, Blob N)
    layer1 = assocR >>> right (mergeLayer >>> connectingLayer >>> sigmoidLayer) >>> lstmLayer

    layer2 :: Layer ((Blob N, Blob N), Blob N) (Blob N, Blob N)
    layer2 = assocR >>> right (mergeLayer >>> connectingLayer >>> sigmoidLayer) >>> lstmLayer

    finalx :: Layer (Blob N) (Blob 256)
    finalx = connectingLayer

    final :: Layer  (Blob N, Int) ()
    final = left finalx >>> softmaxCost

    layer :: Layer ((H,H), Blob 256) ((H,H), Blob N)
    layer = (layer1 >>> dup >>> right dropL) `stack` (layer2 >>> dup >>> right dropL)

    oneofn :: Vector (Blob 256)
    oneofn = V.generate 256 (\i -> blob (replicate i 0 ++ [1] ++ replicate (255 - i) 0))

  print (params layer + params final)

  let Options _ command = opts

  case command of
   Train initpath input savefile logfile -> do
     (initial, p_layer, p_final) <- case initpath of
       -- Just path -> read <$> readFile path
       Just path -> LB.decode <$> LB.readFile path
       Nothing -> (,,)
                  <$> pure unit
                  <*> sampleIO (initialise layer) -- sampleIO (Parameters <$> V.replicateM (params layer) (uniform (-0.08) 0.08)) --
                  <*> sampleIO (initialise final) -- sampleIO (Parameters <$> V.replicateM (params layer) (uniform (-0.08) 0.08)) --

     deepseqM (initial, p_layer, p_final)

     text <- B.readFile input

     running_average <- newIORef 0
     running_count <- newIORef 0

     startTime <- getTime ProcessCPUTime

     logfp <- case logfile of
               Just logfile -> Just <$> openFile logfile WriteMode
               Nothing -> pure Nothing

     let
       α :: Double
       α = 1 - 1 / 50

       chunkSize = 50 :: Int

       tvec = V.fromList (B.unpack text) :: U.Vector Word8
       ovec = V.map (\c -> oneofn V.! (fromIntegral c)) (V.convert tvec) :: Vector (Blob 256)

       source :: IO (Vector (Blob 256), Vector Int)
       source = do s <- sampleIO (uniform 0 (V.length tvec - chunkSize))
                   let
                     input = (oneofn V.! 0) `V.cons` V.slice s (chunkSize-1) ovec
                     output = V.map fromIntegral (V.convert (V.slice s chunkSize tvec))
                   return (input, output)

       save i init p_layer p_final c = do

         when (not (isInfinite c)) $ do
           modifyIORef' running_average (\x -> (α*x + (1 - α)*c))
           modifyIORef' running_count (\x -> (α*x + (1 - α)*1))

         x <- do q <- readIORef running_average
                 w <- readIORef running_count
                 return ((q / w) / fromIntegral (chunkSize - 1))

         when (i `mod` 50 == 0) $ do
           now <- getTime ProcessCPUTime
           let tdiff = fromIntegral (timeSpecAsNanoSecs (now - startTime)) / (10^9) :: Double
           putStrLn $ printf "[% 11.4f]  %i  %f  %f" tdiff i x (c / fromIntegral (chunkSize-1))
           case logfp of
            Just fp -> hPutStrLn fp (printf "%f %i %f" tdiff i x) >> hFlush fp
            Nothing -> return ()
         when (i `mod` 1000 == 0) $ do
           -- writeFile savefile $ show (init, p_layer, p_final)
           case savefile of
            Just savefile -> do
              LB.writeFile (printf "%s-%6.6i-%5.5f.bin" savefile i x) $ LB.encode (init, p_layer, p_final)
              LB.writeFile (savefile ++ "-latest.bin") $ LB.encode (init, p_layer, p_final)
            Nothing -> return ()
           LB.putStrLn . LB.pack . take 100 $ sampleRNN init unit layer p_layer finalx p_final

     deepseqM (tvec, ovec)

     descent initial layer p_layer final p_final source save

   Sample initpath length -> do
     (initial, p_layer, p_final) <- LB.decode <$> LB.readFile initpath
     deepseqM (initial, p_layer, p_final)

     print $ V.maximum (getParameters p_layer)

     let text = sampleRNN initial (oneofn V.! 0) layer p_layer finalx p_final
     LB.putStrLn . LB.pack $ case length of
                 Just n -> take n text
                 Nothing -> text

   CheckDeriv -> do
    checkGradient $ splitLayer >>> left (hierLayerN 4 :: Layer (Blob 50) (Blob 100)) >>> (quadraticCost :: Network Identity (Blob 100, Blob 100) ())

    checkGradient $ splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

    checkGradient $ (fcLayer :: Network Identity (Blob 20) (Blob 20)) >>> splitLayer >>> (quadraticCost :: Network Identity (Blob 10, Blob 10) ())

    checkGradient $ splitLayer >>> (lstmLayer :: Network Identity (Blob 1, Blob 4) (Blob 1, Blob 1)) >>> quadraticCost

    checkGradient $ splitLayer >>> left (freeLayer :: Layer (Blob 50) (Blob 100)) >>> (quadraticCost :: Network Identity (Blob 100, Blob 100) ())

    checkGradient $ splitLayer >>> left (hierLayer :: Layer (Blob 50) (Blob 100)) >>> (quadraticCost :: Network Identity (Blob 100, Blob 100) ())

    checkGradient $ (feedR 7 softmaxCost :: Layer (Blob 10) ())

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
