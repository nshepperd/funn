{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}
module AI.Funn.CL.MonadCL (
  initOpenCL, getPlatformID, getContext, getDeviceID, getCommandQueue,
  KernelArg(..), doubleArg, int32Arg, runKernel
  ) where

import           Control.Applicative
import           Control.Monad
import           Control.Monad.Trans.Free
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Int

import           Control.Concurrent.MVar
import           Data.Proxy
import           Data.Random

import           Control.Lens
import           Control.Monad.IO.Class
import           Control.Monad.Reader
import           Control.Monad.State.Lazy
import           Control.Monad.Trans

import           Data.Map (Map)
import qualified Data.Map.Strict as Map

import           Foreign.Ptr
import           GHC.Float
import           System.IO.Unsafe
import           Data.IORef

import qualified Foreign.OpenCL.Bindings as CL
import qualified Foreign.OpenCL.Bindings.Synchronization as CL

data CLInfo = CLInfo {
  _platformID :: CL.PlatformID,
  _context :: CL.Context,
  _device :: CL.DeviceID,
  _queue :: CL.CommandQueue
  }

global_clinfo :: IORef (Maybe CLInfo)
global_clinfo = unsafePerformIO $ newIORef Nothing

global_kernelcache :: IORef (Map String CL.Kernel)
global_kernelcache = unsafePerformIO $ newIORef Map.empty

clPlatformID :: CL.PlatformID
clPlatformID = case unsafePerformIO (readIORef global_clinfo) of
  Just clinfo -> _platformID clinfo

clContext :: CL.Context
clContext = case unsafePerformIO (readIORef global_clinfo) of
  Just clinfo -> _context clinfo

clDeviceID :: CL.DeviceID
clDeviceID = case unsafePerformIO (readIORef global_clinfo) of
  Just clinfo -> _device clinfo

clCommandQueue :: CL.CommandQueue
clCommandQueue = case unsafePerformIO (readIORef global_clinfo) of
  Just clinfo -> _queue clinfo

initOpenCL :: IO ()
initOpenCL = do
  may <- readIORef global_clinfo
  case may of
    Just _ -> return ()
    Nothing -> do
      clinfo <- do plat:_ <- CL.getPlatformIDs
                   [dev] <- CL.getDeviceIDs [CL.DeviceTypeAll] plat
                   ctx <- CL.createContext [dev] [CL.ContextPlatform plat] CL.NoContextCallback
                   q <- CL.createCommandQueue ctx dev []
                   return (CLInfo plat ctx dev q)
      writeIORef global_clinfo (Just clinfo)

getPlatformID :: (MonadIO m) => m CL.PlatformID
getPlatformID = return clPlatformID

getContext :: (MonadIO m) => m CL.Context
getContext = return clContext

getDeviceID :: (MonadIO m) => m CL.DeviceID
getDeviceID = return clDeviceID

getCommandQueue :: (MonadIO m) => m CL.CommandQueue
getCommandQueue = return clCommandQueue

getKernel :: (MonadIO m) => String -> String -> String -> m CL.Kernel
getKernel key src entryPoint = liftIO $ do
  kmap <- readIORef global_kernelcache
  case Map.lookup key kmap of
    Just kernel -> return kernel
    Nothing -> do
      prog <- CL.createProgram clContext src
      CL.buildProgram prog [clDeviceID] ""
      kernel <- CL.createKernel prog entryPoint
      let
        kmap' = Map.insert key kernel kmap
      return kernel

newtype KernelArg = KernelArg (([CL.KernelArg] -> IO ()) -> IO ())

instance Monoid KernelArg where
  mempty = KernelArg (\k -> k [])
  mappend (KernelArg f1) (KernelArg f2) = KernelArg $ \k ->
    f1 (\args1 ->
          f2 (\args2 -> k (args1 ++ args2)))

doubleArg :: Double -> KernelArg
doubleArg x = KernelArg run
  where
    run f = f [CL.VArg x]

int32Arg :: Integral a => a -> KernelArg
int32Arg a = KernelArg run
  where
    run f = f [CL.VArg (fromIntegral a :: Int32)]

runKernel :: MonadIO m => String -> String  -> [KernelArg] -> [CL.ClSize] -> [CL.ClSize] -> [CL.ClSize] -> m ()
runKernel source entryPoint args globalOffsets globalSizes localSizes =
  do queue <- getCommandQueue
     kernel <- getKernel (source ++ "/" ++ entryPoint) source entryPoint
     let go [] real_args = do
           CL.setKernelArgs kernel (real_args [])
           ev <- CL.enqueueNDRangeKernel queue kernel globalOffsets globalSizes localSizes []
           CL.waitForEvents [ev]
         go (KernelArg f : fs) real_args = f (\arg -> go fs (real_args . (arg++)))
     liftIO $ go args id
