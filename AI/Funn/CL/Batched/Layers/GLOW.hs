{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Batched.Layers.GLOW (reshapeInv, affineCouplingInv, splitChannelInv) where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits
import           System.IO.Unsafe
import           Text.Printf

import           AI.Funn.CL.Batched.BTensor (BTensor(..))
import qualified AI.Funn.CL.Batched.BTensor as BT
import           AI.Funn.CL.Batched.GLOW (Invertible(..), invert)
import           AI.Funn.CL.Batched.Layers.Simple
import           AI.Funn.CL.Batched.Network (Network(..), liftDiff)
import           AI.Funn.CL.Batched.Param (Param(..))
import qualified AI.Funn.CL.Batched.Param as Param
import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import qualified AI.Funn.CL.Layers.Tensor as Layers
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Tensor (Tensor(..), MTensor(..))
import qualified AI.Funn.CL.Tensor as Tensor
import qualified AI.Funn.CL.TensorLazy as TL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import qualified AI.Funn.Diff.Diff as Diff
import qualified AI.Funn.Flat.Blob as Blob
import           AI.Funn.Indexed.Indexed
import           AI.Funn.Space
import           AI.Funn.TypeLits

reshapeInv :: (KnownNat ω, Prod as ~ Prod bs, Monad m)
           => Invertible m ω 0 (Tensor (ω ': as)) (Tensor (ω ': bs))
reshapeInv = Invertible reshapeNet reshapeNet

reshapeInvLazy :: (KnownNat ω, Prod as ~ Prod bs, Monad m)
               => Invertible m ω 0 (TL.Tensor (ω ': as)) (TL.Tensor (ω ': bs))
reshapeInvLazy = Invertible (liftDiff (Diff run)) (liftDiff (Diff run))
  where
    run as = return (TL.reshape as, backward)
    backward bs = return (TL.reshape bs)

splitInvLazy :: (KnownDimsF [ω, a, b], Monad m)
             => Invertible m ω 0 (TL.Tensor [ω, a+b]) (TL.Tensor [ω, a], TL.Tensor [ω, b])
splitInvLazy = Invertible split append
  where
    split = liftDiff (Diff $ \ab -> return (TL.splitW ab, return . uncurry TL.appendW))
    append = liftDiff (Diff $ \(a,b) -> return (TL.appendW a b, return . TL.splitW))

appendInvLazy :: (KnownDimsF [ω, a, b], Monad m)
             => Invertible m ω 0 (TL.Tensor [ω, a], TL.Tensor [ω, b]) (TL.Tensor [ω, a+b])
appendInvLazy = invert splitInvLazy


{-# NOINLINE splitProgram #-}
splitProgram :: KernelProgram '[TensorCL '[a, 2, b], MTensorCL '[2, a, b]]
splitProgram = compile $ \xs ys -> do
  ~[i,a,b] <- traverse get_global_id [0,1,2]
  ys![i,a,b] .= xs![a,i,b]

splitTop :: (KnownNat (Prod ds)) => Tensor (2:ds) -> (Tensor ds, Tensor ds)
splitTop xs = let (ya, yb) = Tensor.split (Tensor.reshape xs)
              in (Tensor.reshape ya, Tensor.reshape yb)

splitChannel :: forall ω w h c. (KnownDimsF [ω, w, h, c])
             => Tensor [ω, w, h, 2*c]
             -> (Tensor [ω, w, h, c], Tensor [ω, w, h, c])
splitChannel xs = unsafePerformIO $ do
  yab <- Tensor.new :: IO (MTensor [2, ω * w * h, c])
  let xss = Tensor.reshape xs :: Tensor [ω * w * h, 2, c]
  clfun splitProgram (dimVal yab) xss yab :: IO ()
  return $ splitTop (Tensor.reshape (Tensor.unsafeFreeze yab))

{-# NOINLINE appendProgram #-}
appendProgram :: KernelProgram '[TensorCL '[a, b], TensorCL '[a, b], MTensorCL '[a, 2, b]]
appendProgram = compile $ \xs ys zs -> do
  ~[a,b] <- traverse get_global_id [0,1]
  zs![a,0,b] .= xs![a,b]
  zs![a,1,b] .= ys![a,b]

appendChannel :: forall ω w h c. (KnownDimsF [ω, w, h, c])
              => Tensor [ω, w, h, c]
              -> Tensor [ω, w, h, c]
              -> Tensor [ω, w, h, 2*c]
appendChannel xa xb = unsafePerformIO $ do
  y <- Tensor.new :: IO (MTensor [ω * w * h, 2, c])
  let xa' = Tensor.reshape xa
      xb' = Tensor.reshape xb
  clfun appendProgram (dimVal xa') xa' xb' y :: IO ()
  return $ (Tensor.reshape (Tensor.unsafeFreeze y))

splitChannelInv :: forall w h c ω m. (KnownDimsF [ω, w, h, c], Monad m)
                => Invertible m ω 0 (Tensor [ω, w, h, 2*c]) (Tensor [ω, w, h, c], Tensor [ω, w, h, c])
splitChannelInv = Invertible fwd bwd
  where
    fwd = liftDiff (Diff runForward)
    runForward ab = let (a, b) = splitChannel ab
                        back (da, db) = return (appendChannel da db)
                    in return ((a, b), back)
    bwd = liftDiff (Diff runBackward)
    runBackward (a, b) = let back ab = return (splitChannel ab)
                         in return (appendChannel a b, back)

splitChannelNet :: forall w h c ω m. (KnownDimsF [ω, w, h, c], Monad m)
                => Network m ω 0 (Tensor [ω, w, h, 2*c]) (Tensor [ω, w, h, c], Tensor [ω, w, h, c])
splitChannelNet = invForward splitChannelInv

-- Calculates ya = exp(s) * xa + t.
affineFwdNet :: forall w h c ω m. (KnownDimsF [ω, w, h, c], Monad m)
             => Network m ω 0 (Tensor [ω, w, h, 2*c], Tensor [ω, w, h, c]) (Tensor [ω, w, h, c])
affineFwdNet = first (invForward splitChannelInv) ~>> liftDiff (Diff run)
  where
    run ((s,t), xa) =
      let ya = affineExp s xa t
          back dya = let (ds, dxa) = affineExpDiff s xa dya
                         dt = dya
                     in return ((ds, dt), dxa)
      in return (ya, back)

{-# NOINLINE affineProgram #-}
affineProgram :: KernelProgram '[TensorCL '[a], TensorCL '[a], TensorCL '[a], MTensorCL '[a]]
affineProgram = compile $ \ss xs ts ys -> do
  ~[i] <- traverse get_global_id [0]
  ys![i] .= exp (ss![i]) * xs![i] + ts![i]

{-# NOINLINE affineDiffProgram #-}
affineDiffProgram :: KernelProgram '[TensorCL '[a], TensorCL '[a], TensorCL '[a],
                                     MTensorCL '[a], MTensorCL '[a]]
affineDiffProgram = compile $ \s xa dya ds dxa -> do
  ~[i] <- traverse get_global_id [0]
  es <- eval $ exp (s![i])
  ds![i] .= es * xa![i] * dya![i]
  dxa![i] .= es * dya![i]

-- Calculates ya = exp(s) * xa + t.
affineExp :: forall ω w h c. (KnownDimsF [ω, w, h, c])
          => Tensor [ω, w, h, c]
          -> Tensor [ω, w, h, c]
          -> Tensor [ω, w, h, c]
          -> Tensor [ω, w, h, c]
affineExp s xa t = unsafePerformIO $ do
  ya <- Tensor.new
  clfun affineProgram (dimVal ya) (Tensor.reshape s) (Tensor.reshape xa) (Tensor.reshape t) ya :: IO ()
  return $ (Tensor.reshape (Tensor.unsafeFreeze ya))

-- Calculates derivatives dL/ds and dL/dxa for ya = exp(s) * xa + t.
affineExpDiff :: forall ω w h c. (KnownDimsF [ω, w, h, c])
              => Tensor [ω, w, h, c]
              -> Tensor [ω, w, h, c]
              -> Tensor [ω, w, h, c]
              -> (Tensor [ω, w, h, c], Tensor [ω, w, h, c])
affineExpDiff s xa dya = unsafePerformIO $ do
  ds <- Tensor.new
  dxa <- Tensor.new
  clfun affineDiffProgram (dimVal ds) (Tensor.reshape s) (Tensor.reshape xa) (Tensor.reshape dya) ds dxa
  return (Tensor.reshape (Tensor.unsafeFreeze ds),
          Tensor.reshape (Tensor.unsafeFreeze dxa))

{-# NOINLINE affineInvertProgram #-}
affineInvertProgram :: KernelProgram '[TensorCL '[a], TensorCL '[a], TensorCL '[a], MTensorCL '[a]]
affineInvertProgram = compile $ \s ya t xa -> do
  ~[i] <- traverse get_global_id [0]
  xa![i] .= (ya![i] - t![i]) / exp (s![i])

-- Calculates xa = (ya - t) / exp(s)
affineInvert :: forall ω w h c. (KnownDimsF [ω, w, h, c])
             => Tensor [ω, w, h, c]
             -> Tensor [ω, w, h, c]
             -> Tensor [ω, w, h, c]
             -> Tensor [ω, w, h, c]
affineInvert s ya t = unsafePerformIO $ do
  xa <- Tensor.new
  clfun affineInvertProgram (dimVal xa) (Tensor.reshape s) (Tensor.reshape ya) (Tensor.reshape t) xa :: IO ()
  return $ (Tensor.reshape (Tensor.unsafeFreeze xa))

{-# NOINLINE affineInvertDiffProgram #-}
affineInvertDiffProgram :: KernelProgram '[TensorCL '[a], TensorCL '[a], TensorCL '[a],
                                           MTensorCL '[a], MTensorCL '[a], MTensorCL '[a]]
affineInvertDiffProgram = compile $ \s xa dxa ds dya dt -> do
  ~[i] <- traverse get_global_id [0]
  -- dxa/ds = -(ya - t) / exp(s) = -xa
  -- dxa/dya = 1 / exp(s)
  -- dxa/dt = -1 / exp(s) = -dya
  ds![i] .= -xa![i] * dxa![i]
  dya![i] .= dxa![i] / exp (s![i])
  dt![i] .= -dya![i]

-- Calculates dL/ds, dL/dya, dL/dt from (s, ya, t, dxa)
affineInvertDiff :: forall ω w h c. (KnownDimsF [ω, w, h, c])
                 => Tensor [ω, w, h, c]
                 -> Tensor [ω, w, h, c]
                 -> Tensor [ω, w, h, c]
                 -> (Tensor [ω, w, h, c], Tensor [ω, w, h, c], Tensor [ω, w, h, c])
affineInvertDiff s xa dxa = unsafePerformIO $ do
  dya <- Tensor.new
  dt <- Tensor.new
  ds <- Tensor.new
  clfun affineInvertDiffProgram (dimVal dya) (Tensor.reshape s) (Tensor.reshape xa) (Tensor.reshape dxa) ds dya dt
  return $ (Tensor.reshape (Tensor.unsafeFreeze ds),
            Tensor.reshape (Tensor.unsafeFreeze dya),
            Tensor.reshape (Tensor.unsafeFreeze dt))

affineCouplingInv :: forall w h c ω p m. (KnownDimsF [ω, w, h, c, p], MonadIO m)
                  => Network m ω p (Tensor [ω, w, h, c]) (Tensor [ω, w, h, 2*c])
                  -> Invertible m ω p (Tensor [ω, w, h, 2*c]) (Tensor [ω, w, h, 2*c])
affineCouplingInv net = Invertible fwd bwd
  where
    fwd = Network (Diff runForward) (netInit net)
    runForward (par, xab) = do
      let (xa, xb) = splitChannel xab
      ((s,t), k) <- runDiff (netDiff $ net ~>> splitChannelNet) (par, xb)
      let ya = affineExp s xa t
          yb = xb
          yab = appendChannel yb ya
          backward dy = do
            let (dyb, dya) = splitChannel dy
                dt = dya
                (ds, dxa) = affineExpDiff s xa dya
            (dpar, dxb_nn) <- k (ds, dt)
            dxb <- dyb `plus` dxb_nn
            let dxab = appendChannel dxa dxb
            return (dpar, dxab)
      return (yab, backward)

    bwd = Network (Diff runBackward) (netInit net)
    runBackward (par, yab) = do
      let (yb, ya) = splitChannel yab
          xb = yb
      ((s,t), k) <- runDiff (netDiff $ net ~>> splitChannelNet) (par, xb)
      let xa = affineInvert s ya t
          xab = appendChannel xa xb
          backward dxab = do
            let (dxa, dxb_pass) = splitChannel dxab
                (ds, dya, dt) = affineInvertDiff s xa dxa
            (dpar, dxb_nn) <- k (ds, dt)
            dxb <- plus dxb_pass dxb_nn
            let dyb = dxb
                dyab = appendChannel dyb dya
            return (dpar, dyab)
      return (xab, backward)
