SELECT H.playerID, H.yearid, H.ballots, H.needed, H.votes,
    M.weight, M.height,
    (SELECT sum(b.G) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as games,
    (SELECT sum(b.AB) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as atBats,
    (SELECT sum(b.R) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as runs,

(SELECT sum(b.H) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as hits,
    (SELECT sum(b.`2B`) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as doubles,
    (SELECT sum(b.`3B`) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as triples,
 (SELECT sum(b.HR) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as homeruns,
    (SELECT sum(b.RBI) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as RBIs,
    (SELECT sum(b.SB) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as stolenBases,
    
    (SELECT sum(b.CS) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as caughtStealings,
    (SELECT sum(b.BB) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as baseonballs,
    (SELECT sum(b.SO) FROM Batting as b WHERE H.playerID=b.playerID AND H.yearID>=b.yearID) as strikeouts,
    H.inducted
from HallOfFame as H
inner join Master as M ON(H.playerID=M.playerID)
group by H.playerID, H.yearID

