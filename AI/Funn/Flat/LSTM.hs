{-# LANGUAGE NoStarIsType #-}
{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.Flat.LSTM (lstmDiff) where

import           GHC.TypeLits

import           Control.Applicative
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Proxy
import           Data.Random

import           Control.DeepSeq
import qualified Data.Vector.Generic as V

import qualified Data.Vector.Storable as S
import qualified Data.Vector.Storable.Mutable as M
import           Foreign.C
import           Foreign.Ptr
import           System.IO.Unsafe

import           AI.Funn.Diff.Diff (Derivable(..), Additive(..), Diff(..))
import           AI.Funn.Flat.Blob (Blob(..), blob, getBlob)
import qualified AI.Funn.Flat.Blob as Blob

foreign import ccall "lstm_forward" lstmForwardFFI :: CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "lstm_backward" lstmBackwardFFI :: CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE lstmForward #-}
lstmForward :: Int -> S.Vector Double -> S.Vector Double -> S.Vector Double ->
               (S.Vector Double, S.Vector Double, S.Vector Double)
lstmForward n ps hs xs = unsafePerformIO $ do target_hs <- M.replicate n 0 :: IO (M.IOVector Double)
                                              target_ys <- M.replicate n 0 :: IO (M.IOVector Double)
                                              target_store <- M.replicate (8*n) 0 :: IO (M.IOVector Double)
                                              S.unsafeWith hs $ \hbuf -> do
                                                S.unsafeWith ps $ \pbuf -> do
                                                  S.unsafeWith xs $ \xbuf -> do
                                                    M.unsafeWith target_hs $ \thbuf -> do
                                                      M.unsafeWith target_ys $ \tybuf -> do
                                                        M.unsafeWith target_store $ \tsbuf -> do
                                                          lstmForwardFFI (fromIntegral n) pbuf hbuf xbuf thbuf tybuf tsbuf
                                              new_hs <- V.unsafeFreeze target_hs
                                              new_ys <- V.unsafeFreeze target_ys
                                              store <- V.unsafeFreeze target_store
                                              return (new_hs, new_ys, store)

{-# NOINLINE lstmBackward #-}
lstmBackward :: Int -> S.Vector Double -> S.Vector Double -> S.Vector Double -> S.Vector Double ->
                (S.Vector Double, S.Vector Double, S.Vector Double)
lstmBackward n ps store delta_h delta_y = unsafePerformIO $ do
  target_d_ws <- M.replicate (2*n) 0 :: IO (M.IOVector Double)
  target_d_hs <- M.replicate n 0 :: IO (M.IOVector Double)
  target_d_xs <- M.replicate (4*n) 0 :: IO (M.IOVector Double)
  S.unsafeWith ps $ \pbuf -> do
    S.unsafeWith store $ \sbuf -> do
      S.unsafeWith delta_h $ \dbuf -> do
        S.unsafeWith delta_y $ \dybuf -> do
          M.unsafeWith target_d_ws $ \dwbuf -> do
            M.unsafeWith target_d_hs $ \dhbuf -> do
              M.unsafeWith target_d_xs $ \dxbuf -> do
                lstmBackwardFFI (fromIntegral n) pbuf sbuf dbuf dybuf dwbuf dhbuf dxbuf
  d_ws <- V.unsafeFreeze target_d_ws
  d_hs <- V.unsafeFreeze target_d_hs
  d_xs <- V.unsafeFreeze target_d_xs
  return (d_ws, d_hs, d_xs)

lstmDiff :: forall n m. (Monad m, KnownNat n) => Diff m (Blob (2*n), (Blob n, Blob (4*n))) (Blob n, Blob n)
lstmDiff = Diff run
  where
    run (par, (hidden, inputs)) =
      let (new_h, new_y, store) = (lstmForward n (getBlob par)
                                   (getBlob hidden)
                                   (getBlob inputs))
          backward (dh, dy) = let (d_ws, d_hs, d_xs) = (lstmBackward n (getBlob par)
                                                        store (getBlob dh) (getBlob dy))
                              in return (blob d_ws, (blob d_hs, blob d_xs))
      in return ((blob new_h, blob new_y), backward)

    n :: Int
    n = fromIntegral (natVal (Proxy :: Proxy n))
