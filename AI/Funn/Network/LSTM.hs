{-# LANGUAGE TypeFamilies, KindSignatures, DataKinds, TypeOperators #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
module AI.Funn.Network.LSTM (lstmLayer) where

import           GHC.TypeLits

import           Control.Applicative
import           Data.Foldable
import           Data.Traversable
import           Data.Monoid
import           Data.Proxy
import           Data.Random

import           Control.DeepSeq
import qualified Data.Vector.Generic as V

import           AI.Funn.Diff.Diff (Derivable(..), Additive(..), Diff(..))
import           AI.Funn.Flat.Blob (Blob(..))
import qualified AI.Funn.Flat.Blob as Blob
import qualified AI.Funn.Flat.LSTM as Flat
import AI.Funn.Network.Network

lstmLayer :: forall n m. (Monad m, KnownNat n) => Network m ((Blob n, Blob (4*n))) (Blob n, Blob n)
lstmLayer = Network Proxy Flat.lstmDiff initial
  where
    initial = Blob.generate (pure 1)
