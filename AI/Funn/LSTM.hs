{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE ForeignFunctionInterface #-}
module AI.Funn.LSTM where

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

import           AI.Funn.Network
import           AI.Funn.Flat

foreign import ccall "lstm_forward" lstmForwardFFI :: CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()
foreign import ccall "lstm_backward" lstmBackwardFFI :: CInt -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> Ptr Double -> IO ()

{-# NOINLINE lstmForward #-}
lstmForward :: Int -> S.Vector Double -> S.Vector Double -> S.Vector Double -> (S.Vector Double, S.Vector Double, S.Vector Double)
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
lstmBackward :: Int -> S.Vector Double -> S.Vector Double -> S.Vector Double -> S.Vector Double -> (S.Vector Double, S.Vector Double, S.Vector Double)
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

lstmLayer :: forall n m. (Monad m, KnownNat n) => Network m (Blob n, Blob (4*n)) (Blob n, Blob n)
lstmLayer = Network eval numpar init
  where
    eval par (hidden, inputs) = let (new_h, new_y, store) = (lstmForward n (getParameters par)
                                                             (getBlob hidden)
                                                             (getBlob inputs))
                                    -- !_ = if any isNaN (V.toList new_h) then
                                    --        error ("NaN: new_h " ++ show (par, hidden, inputs))
                                    --      else ()
                                    -- !_ = if any isNaN (V.toList new_y) then
                                    --        error ("NaN: new_y " ++ show (par, hidden, inputs))
                                    --      else ()
                                    -- !_ = if any isNaN (V.toList store) then
                                    --        error ("NaN: store " ++ show (par, hidden, inputs))
                                    --      else ()
                                    backward (dh, dy) = let (d_ws, d_hs, d_xs) = (lstmBackward n (getParameters par)
                                                                                    store (getBlob dh) (getBlob dy))
                                                        in return ((Blob d_hs, Blob d_xs), [Parameters d_ws])
                                in return ((Blob new_h, Blob new_y), 0, backward)
    numpar = 2*n
    init = return (Parameters (V.replicate (2*n) 1))

    n :: Int
    n = fromIntegral (natVal (Proxy :: Proxy n))
