"""Write the list of match ids to download: all Barcelona La Liga matches + all 2015/16 matches."""
import json, os, sys
mdir, out = sys.argv[1], sys.argv[2]
ids = set()
for f in sorted(os.listdir(mdir)):
    if not f.endswith('.json'):
        continue
    ms = json.load(open(os.path.join(mdir, f)))
    sid = int(f.split('_')[1].split('.')[0])
    for m in ms:
        teams = (m['home_team']['home_team_name'], m['away_team']['away_team_name'])
        if sid == 27 or 'Barcelona' in teams:
            ids.add(m['match_id'])
open(out, 'w').write('\n'.join(str(i) for i in sorted(ids)) + '\n')
print(len(ids), 'ids')
