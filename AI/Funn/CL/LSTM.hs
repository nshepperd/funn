{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TypeApplications #-}
{-# OPTIONS_GHC -fplugin GHC.TypeLits.KnownNat.Solver #-}
{-# OPTIONS_GHC -fconstraint-solver-iterations=10 #-}
module AI.Funn.CL.LSTM (lstmDiff) where

import           Control.Applicative
import           Control.Monad
import           Control.Monad.IO.Class
import           Data.Proxy

import           GHC.TypeLits

import           AI.Funn.SomeNat
import           AI.Funn.CL.Blob
import qualified AI.Funn.CL.Blob as Blob
import           AI.Funn.CL.MonadCL
import           AI.Funn.Diff.Diff (Derivable(..), Diff(..))
import           AI.Funn.CL.Code as C

lstmDiff :: forall n m a. (MonadIO m, KnownNat n, CLNum a)
         => Diff m
            (Blob a (2*n), (Blob a n, Blob a (4*n)))
            (Blob a n, Blob a n)
lstmDiff = Diff run
  where
    run (ws, (cs, xs)) = do
      (store :: MBlob a (8 * n)) <- createBlob @ (8 * n)
      new_cs <- createBlob
      ys <- createBlob
      (runKernel forwardSrc "run"
       [blobArg ws, blobArg cs, blobArg xs, blobArg store, blobArg new_cs, blobArg ys]
       [] [fromIntegral n] [])
      frozen_store <- unsafeFreeze store
      frozen_new_cs <- unsafeFreeze new_cs
      frozen_ys <- unsafeFreeze ys
      return ((frozen_new_cs, frozen_ys), backward ws frozen_store)

    backward ws store (dcs', dys) = do
      dws <- createBlob
      dcs <- createBlob
      dxs <- createBlob
      (runKernel backwardSrc "run"
       [blobArg ws, blobArg store, blobArg dcs', blobArg dys,
        blobArg dws, blobArg dcs, blobArg dxs]
       [] [fromIntegral n] [])

      frozen_dws <- unsafeFreeze dws
      frozen_dcs <- unsafeFreeze dcs
      frozen_dxs <- unsafeFreeze dxs
      return (frozen_dws, (frozen_dcs, frozen_dxs))


    forwardSrc = C.kernel forwardKernel
    forwardKernel :: ArrayR a -> ArrayR a -> ArrayR a
                  -> ArrayW a -> ArrayW a -> ArrayW a
                  -> CL ()
    forwardKernel ws cs xs store cs' ys = do
      i <- get_global_id 0
      in_index <- eval (4 * i)
      ws_index <- eval (2 * i)
      let
        x = 2 * (at xs in_index) - 1
        gate_input = at xs (in_index+1)
        gate_forget = at xs (in_index+2)
        gate_output = at xs (in_index+3)
        cell_old = at cs i
        ws_keep = ws `at` ws_index
        ws_add = ws `at` (ws_index + 1)

      cell_add <- eval $ x * gate_input
      cell_keep <- eval $ cell_old * gate_forget
      cell_new <- eval $ cell_keep * ws_keep + cell_add * ws_add

      let
        y = cell_new * gate_output

      store_index <- eval (8 * i)
      at store (store_index) .= x
      at store (store_index + 1) .= gate_input
      at store (store_index + 2) .= gate_forget
      at store (store_index + 3) .= gate_output
      at store (store_index + 4) .= cell_old
      at store (store_index + 5) .= cell_add
      at store (store_index + 6) .= cell_keep
      at store (store_index + 7) .= cell_new

      at cs' i .= cell_new
      at ys i .= y

    backwardSrc = C.kernel backwardKernel
    backwardKernel :: ArrayR a -> ArrayR a -> ArrayR a -> ArrayR a
                   -> ArrayW a -> ArrayW a -> ArrayW a
                   -> CL ()
    backwardKernel ws store dcs' dys out_dws out_dcs out_dxs = do
      i <- get_global_id 0
      in_index <- eval (4 * i)
      ws_index <- eval (2 * i)
      store_index <- eval (8 * i)
      let
        x = store `at` store_index
        gate_input = store `at` (store_index + 1)
        gate_forget = store `at` (store_index + 2)
        gate_output = store `at` (store_index + 3)
        cell_old = store `at` (store_index + 4)
        cell_add = store `at` (store_index + 5)
        cell_keep = store `at` (store_index + 6)
        cell_new = store `at` (store_index + 7)

        ws_keep = ws `at` (ws_index + 0)
        ws_add = ws `at` (ws_index + 1)

        dy = dys `at` i
        dc' = dcs' `at` i

      dcell_new <- eval $ dy * gate_output + dc'
      dcell_keep <- eval $ dcell_new * ws_keep
      dcell_add <- eval $ dcell_new * ws_add

      let
        dws_keep = dcell_new * cell_keep
        dws_add = dcell_new * cell_add
        dc = dcell_keep * gate_forget
        dx = dcell_add * gate_input
        dgate_input = dcell_add * x
        dgate_forget = dcell_keep * cell_old
        dgate_output = dy * cell_new

      out_dws `at` ws_index .= dws_keep
      out_dws `at` (ws_index + 1) .= dws_add
      out_dcs `at` i .= dc
      out_dxs `at` in_index .= 2 * dx
      out_dxs `at` (in_index + 1) .= dgate_input
      out_dxs `at` (in_index + 2) .= dgate_forget
      out_dxs `at` (in_index + 3) .= dgate_output

    n = natVal (Proxy @ n)
