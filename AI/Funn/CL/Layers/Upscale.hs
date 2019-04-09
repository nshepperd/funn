{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE UndecidableSuperClasses #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.Normalise #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=20 #-}
module AI.Funn.CL.Layers.Upscale (doubleDiff) where

import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Foldable
import           Data.List
import           Data.Monoid
import           Data.Proxy
import           Data.Traversable
import           GHC.TypeLits

import qualified AI.Funn.CL.DSL.AST as AST
import           AI.Funn.CL.DSL.Array
import           AI.Funn.CL.DSL.Code as C
import           AI.Funn.CL.DSL.Tensor
import           AI.Funn.CL.Function
import           AI.Funn.CL.MonadCL
import           AI.Funn.CL.Network
import           AI.Funn.CL.Tensor (Tensor, MTensor)
import qualified AI.Funn.CL.Tensor as Tensor
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.Space
import           AI.Funn.TypeLits


{-# NOINLINE doubleForward #-}
doubleForward :: KernelProgram '[TensorCL [w, h, c],
                                 MTensorCL [2*w, 2*h, c]]
doubleForward = compile $ \input output -> do
  ~[ox, oy, c] <- traverse get_global_id [0, 1, 2]
  output![ox, oy, c] .= input![ox `div'` 2, oy `div'` 2, c]

{-# NOINLINE doubleBackward #-}
doubleBackward :: KernelProgram '[MTensorCL [w, h, c],
                                  TensorCL [2*w, 2*h, c]]
doubleBackward = compile $ \d_input d_output -> do
  ~[ix, iy, c] <- traverse get_global_id [0, 1, 2]
  ox <- eval (2 * ix)
  oy <- eval (2 * iy)
  d_input![ix, iy, c] .= sum [d_output![x,y,c] | (x,y) <- [
                                 (ox,   oy),
                                 (ox+1, oy),
                                 (ox,   oy+1),
                                 (ox+1, oy+1)]]

doubleDiff :: forall w h c m.
              (MonadIO m, KnownDimsF [w, h, c])
           => Diff m (Tensor [w, h, c]) (Tensor [2*w, 2*h, c])
doubleDiff = Diff run
  where
    run input = do
      output <- Tensor.new
      liftIO (clfun doubleForward (dimVal output) input output :: IO ())
      return (Tensor.unsafeFreeze output, backward)

    backward d_output = do
      d_input <- Tensor.new
      liftIO (clfun doubleBackward (dimVal d_input) d_input d_output :: IO ())
      return (Tensor.unsafeFreeze d_input)

doubleNet :: (MonadIO m, KnownDimsF [w, h, c])
          => Network m 0 (Tensor [w, h, c]) (Tensor [2*w, 2*h, c])
doubleNet = liftDiff doubleDiff
