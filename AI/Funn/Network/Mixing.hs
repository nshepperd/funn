{-# LANGUAGE TypeFamilies, DataKinds #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
module AI.Funn.Network.Mixing (amixLayer) where

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
import qualified AI.Funn.Flat.Mixing as Diff
import           AI.Funn.Network.Network

amixLayer :: forall s a b m. (Monad m, KnownNat s, KnownNat a, KnownNat b) =>
            Proxy s -> Network m (Blob a) (Blob b)
amixLayer proxy = Network Proxy (Diff.amixDiff proxy) initial
  where
    -- Could probably do better than this...
    initial = Blob.generate (normal 0 (1 / sqrt (fromIntegral s)))
    s = natVal (Proxy @ s)
