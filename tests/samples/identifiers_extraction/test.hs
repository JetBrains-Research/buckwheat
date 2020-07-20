lookupFunc :: String -> Reader DefsM E
lookupFunc x = R.ask >>= pure . (M.! x)

matchPat :: VarsM -> (Pat, E) -> Maybe VarsM
matchPat m (VarPat x, e)       = Just $ M.insert x e m
matchPat m (BPat b, B b')      = if b == b' then Just m else Nothing
matchPat m (CPat cp ps, C c s) = do
    when (length ps /= length s || cp /= c) $ fail "Can't pattern match."
    foldM matchPat m $ zip ps s
matchPat _ _              = Nothing
