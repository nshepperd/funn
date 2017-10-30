{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Flat (
  reluDiff, sigmoidDiff, tanhDiff,
  fcDiff, quadraticCost, splitDiff, mergeDiff,
  softmaxCost
  ) where

import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Proxy
import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S
import           Debug.Trace
import qualified Foreign.OpenCL.Bindings as CL
import           GHC.Float
import           GHC.TypeLits
import           System.IO.Unsafe

import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.Space
import           AI.Funn.TypeLits

data KName = ReluF Precision
           | ReluB Precision
           | SigmoidF Precision
           | SigmoidB Precision
           | TanhF Precision
           | TanhB Precision
           | FCForward Precision
           | FCBackWS Precision
           | FCBackXS Precision
           | FCBackBS Precision
  deriving (Show, Eq, Ord)

{-# NOINLINE memoTable #-}
memoTable :: KTable KName
memoTable = newKTable unsafePerformIO

reluDiff :: forall n m a. (MonadIO m, KnownNat n, Relational a, CLFloats a) => Diff m (Blob a n) (Blob a n)
reluDiff = Diff run
  where
    run xs = do
      ys <- relu xs
      return (ys, reluBack xs)

    relu = mapBlob' memoTable (ReluF (precision @a)) (\x -> fmax 0 x)
    reluBack = zipWithBlob' memoTable (ReluB (precision @a)) (\x dy -> fstep 0 x * dy)

sigmoidDiff :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Diff m (Blob a n) (Blob a n)
sigmoidDiff = Diff run
  where
    run xs = do
      ys <- sigmoid xs
      return (ys, sigmoidBack xs)

    sigmoid = mapBlob memoTable (SigmoidF (precision @a)) $ \x -> do
      z <- eval (exp x)
      return $ z / (1 + z)

    sigmoidBack = zipWithBlob memoTable (SigmoidB (precision @a)) $ \x dy -> do
      z <- eval $ exp (-abs x)
      return $ dy * z / (1 + z)^2

tanhDiff :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Diff m (Blob a n) (Blob a n)
tanhDiff = Diff run
  where
    run xs = do
      ys <- tanhForward xs
      return (ys, tanhBack xs)

    tanhForward = mapBlob memoTable (TanhF (precision @a)) $ \x -> do
      zp <- eval (exp x)
      zm <- eval (exp (-x))
      return $ (zp - zm) / (zp + zm)

    tanhBack = zipWithBlob memoTable (TanhB (precision @a)) $ \x dy -> do
      zp <- eval $ exp x
      zm <- eval $ exp (-x)
      z <- eval $ 2 / (zp + zm)
      return $ dy * z^2

quadraticCost :: forall n m a. (MonadIO m, KnownNat n, CLFloats a) => Diff m (Blob a n, Blob a n) Double
quadraticCost = Diff run
  where
    run (xs, ys) = do
      ds <- subBlob xs ys
      rs <- Blob.toList ds
      let o = sum [x^2 | x <- rs]
      return (o, backward ds)

    backward ds δ = do
      dx <- scaleBlob (2 * δ) ds
      dy <- scaleBlob (-2 * δ) ds
      return (dx, dy)

softmaxCost :: (MonadIO m, KnownNat n, CLFloats a, Floats a) => Diff m (Blob a n, Int) Double
softmaxCost = Diff run
  where run (bo, target) = do
          oslist <- Blob.toList bo
          let os = V.fromList oslist :: S.Vector Double
              xt = os V.! target
              exp_total_minus_xt = V.imap (\j x -> if j /= target then exp (x - xt) else 0) os
              log_total_minus_xt = log1p (V.sum exp_total_minus_xt)
              cost = log_total_minus_xt
          return (cost, backward target os)

        backward target os dcost = do
          let
            total_but_target = V.sum (V.imap (\i x -> if i /= target then exp x else 0) os)
            total = V.sum (V.map exp os)
            back = V.imap (\j x -> dcost * (if target == j then
                                              -total_but_target / total
                                             else
                                               exp x / total)) os
          backblob <- Blob.fromList (V.toList back)
          return (backblob, ())

fcDiff :: forall α β a m. (MonadIO m, KnownNat α, KnownNat β, CLFloats a) =>
          Diff m (Blob a (α * β + β), Blob a α) (Blob a β)
fcDiff = Diff run
  where
    run (pars, xs) = do
      ys <- createBlob
      liftIO $ forwardK [β] α β pars xs ys
      frozen_ys <- unsafeFreeze ys
      return (frozen_ys, backward pars xs)

    backward pars xs dys = do
      dpars <- createBlob
      dxs <- createBlob
      let (dws :: MBlob a (α * β), dbs :: MBlob a β) = splitBlob dpars
      liftIO $ do
        backwardwsK [α, β] α xs dys dws
        backwardxsK [α] α β pars dys dxs
        backwardbsK [β] dys dbs
      frozen_dpars <- unsafeFreeze dpars
      frozen_dxs <- unsafeFreeze dxs
      return (frozen_dpars, frozen_dxs)

    kahan :: Expr a -> (Expr Int -> Expr a) -> Expr Int -> CL (Expr a)
    kahan initial inputs count = do
      total <- eval initial
      comp <- eval 0
      forEach 0 count $ \x -> do
        input <- eval (inputs x)
        add <- eval (input - comp)
        t <- eval (total + add)
        comp .= (t - total) - add;
        total .= t;
      return total

    forwardK :: [Int] -> Int -> Int -> Blob a (α * β + β) -> Blob a α -> MBlob a β -> IO ()
    forwardK = memoc memoTable (FCForward (precision @a)) forwardSrc
    forwardSrc :: Expr Int -> Expr Int -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    forwardSrc α β pars xs ys = do
      y <- get_global_id 0
      let
        inputs x = (pars `at` (α * y + x)) * (xs `at` x)
      total <- kahan (pars `at` (α*β + y)) inputs α
      at ys y .= total

    backwardwsK :: [Int] -> Int -> Blob a α -> Blob a β -> MBlob a (α * β) -> IO ()
    backwardwsK = memoc memoTable (FCBackWS (precision @a)) backwardwsSrc
    backwardwsSrc :: Expr Int -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    backwardwsSrc α xs dys dws = do
      x <- get_global_id 0
      y <- get_global_id 1
      at dws (y * α + x) .= (xs `at` x) * (dys `at` y)

    backwardxsK :: [Int] -> Int -> Int -> Blob a (α * β + β) -> Blob a β -> MBlob a α -> IO ()
    backwardxsK = memoc memoTable (FCBackXS (precision @a)) backwardxsSrc
    backwardxsSrc :: Expr Int -> Expr Int -> ArrayR a -> ArrayR a -> ArrayW a -> CL ()
    backwardxsSrc α β pars dys dxs = do
      x <- get_global_id 0
      let
        inputs y = at pars (α * y + x) * at dys y
      total <- kahan 0 inputs β
      at dxs x .= total

    backwardbsK :: [Int] -> Blob a β -> MBlob a β -> IO ()
    backwardbsK = memoc memoTable (FCBackBS (precision @a)) backwardbsSrc
    backwardbsSrc :: ArrayR a -> ArrayW a -> CL ()
    backwardbsSrc dys dbs = do
      y <- get_global_id 0
      at dbs y .= at dys y

    α = fromIntegral $ natVal (Proxy :: Proxy α)
    β = fromIntegral $ natVal (Proxy :: Proxy β)

splitDiff :: forall α β a m. (MonadIO m, KnownNat α, KnownNat β, CLFloats a) =>
             Diff m (Blob a (α + β)) (Blob a α, Blob a β)
splitDiff = Diff run
  where
    run ab = pure (splitBlob ab, backward)
    backward (da, db) = catBlob da db

mergeDiff :: forall α β a m. (MonadIO m, KnownNat α, KnownNat β, CLFloats a) =>
             Diff m (Blob a α, Blob a β) (Blob a (α + β))
mergeDiff = Diff run
  where
    run (a,b) = do ab <- catBlob a b
                   pure (ab, backward)
    backward dab = pure (splitBlob dab)
