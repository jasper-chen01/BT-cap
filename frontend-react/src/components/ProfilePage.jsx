import React, { useMemo, useState } from 'react';
import { User } from 'lucide-react';
import Card from './ui/Card';
import Input from './ui/Input';

const ProfilePage = ({ user }) => {
  const [showProfile, setShowProfile] = useState(true);

  const defaults = useMemo(
    () => ({
      name: user?.name || 'Researcher',
      email: user?.email || 'researcher@institution.edu',
      university: user?.university || 'University of Example',
      department: user?.department || 'Neuroscience',
      lab: user?.lab || 'Glioma Systems Lab',
      role: user?.role || 'Principal Investigator',
      orcid: user?.orcid || '0000-0002-1825-0097',
      focus: user?.focus || 'Single-cell glioma characterization',
    }),
    [user]
  );
  const [profile, setProfile] = useState(defaults);

  return (
    <div className="w-full max-w-6xl mx-auto animate-in fade-in duration-500 py-8 px-4 md:px-8 space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-white">Profile</h2>
        <p className="text-slate-400">Manage your research identity and preferences</p>
      </div>

      <Card className="overflow-hidden">
        <button
          type="button"
          onClick={() => setShowProfile((prev) => !prev)}
          className="w-full flex items-center justify-between px-6 py-4 text-left text-slate-200 hover:bg-slate-800/60 transition-colors"
          aria-expanded={showProfile}
        >
          <div className="flex items-center gap-3">
            <User className="text-indigo-400" size={18} />
            <span className="font-semibold">Research Profile</span>
          </div>
          <span className="text-sm text-slate-400">{showProfile ? 'Hide' : 'Show'}</span>
        </button>

        {showProfile && (
          <div className="px-6 pb-6 grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Full Name"
              value={profile.name}
              onChange={(event) => setProfile({ ...profile, name: event.target.value })}
            />
            <Input
              label="Email Address"
              type="email"
              value={profile.email}
              onChange={(event) => setProfile({ ...profile, email: event.target.value })}
            />
            <Input
              label="University"
              value={profile.university}
              onChange={(event) => setProfile({ ...profile, university: event.target.value })}
            />
            <Input
              label="Department"
              value={profile.department}
              onChange={(event) => setProfile({ ...profile, department: event.target.value })}
            />
            <Input
              label="Lab / Research Group"
              value={profile.lab}
              onChange={(event) => setProfile({ ...profile, lab: event.target.value })}
            />
            <Input
              label="Role"
              value={profile.role}
              onChange={(event) => setProfile({ ...profile, role: event.target.value })}
            />
            <Input
              label="ORCID"
              value={profile.orcid}
              onChange={(event) => setProfile({ ...profile, orcid: event.target.value })}
            />
            <Input
              label="Research Focus"
              value={profile.focus}
              onChange={(event) => setProfile({ ...profile, focus: event.target.value })}
            />
          </div>
        )}
      </Card>

    </div>
  );
};

export default ProfilePage;

