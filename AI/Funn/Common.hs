module AI.Funn.Common where

import qualified Data.Vector.Generic as V
import qualified Data.Vector.Storable as S

import qualified Data.Binary as B
import qualified Data.Binary.Put as B
import qualified Data.Binary.Get as B
import           Data.ReinterpretCast

putDouble :: Double -> B.Put
putDouble x = B.putWord64host (doubleToWord x)
getDouble :: B.Get Double
getDouble = wordToDouble <$> B.getWord64host

getVector :: (V.Vector v a) => B.Get a -> B.Get (v a)
getVector g = do size <- B.get
                 V.replicateM (size::Int) g

putVector :: (V.Vector v a) => (a -> B.Put) -> v a -> B.Put
putVector p u = do B.put (V.length u :: Int)
                   V.mapM_ p u


class CheckNAN a where
  check :: (Show b) => String -> a -> b -> ()

instance (S.Storable a, RealFloat a) => CheckNAN (S.Vector a) where
  check s xs b = if V.any (\x -> isNaN x || isInfinite x) xs then
                   error ("[" ++ s ++ "] checkNaN -- " ++ show b)
                 else ()

instance (CheckNAN a, CheckNAN b) => CheckNAN (a,b) where
  check s (x,y) b = check s x b `seq` check s y b
