bl_info = {
    "name": "GMM scene utils",
    "category": "Object",
}

import bpy
from bpy.props import (
        BoolProperty,
        FloatProperty,
        StringProperty,
        EnumProperty,
        )
from bpy_extras.io_utils import (
        ExportHelper)



def hsva_to_rgba(h, s, v, a):
    h_i = int(round(h*6))
    f = h*6 - h_i
    p = v * (1 - s)
    q = v * (1 - f*s)
    t = v * (1 - (1 - f) * s)
    if h_i==0:
        return [v, t, p, a]
    elif h_i==1:
        return [q, v, p, a]
    elif h_i==2:
        return [p, v, t, a]
    elif h_i==3:
        return [p, q, v, a]
    elif h_i==4:
        return [t, p, v, a]
    elif h_i==5:
        return [v, p, q, a]
    return [1,1,1,a]

def indent(str, level):
    return str.replace('\t', '  '*level)

class ExportScene(bpy.types.Operator, ExportHelper):
    """My Object Moving Script"""      # blender will use this as a tooltip for menu items and buttons.
    bl_idname = "export_scene.gmm"        # unique identifier for buttons and menu items to reference.
    bl_label = "Tag object as GMM-Object"         # display name in the interface.
    bl_options = {'PRESET'}  # enable undo for the operator.

    filename_ext = ".yaml"
    filter_glob = StringProperty(
            default="*.obj;*.mtl",
            options={'HIDDEN'},
            )

    use_gmm = BoolProperty(
            name="Write GMMs",
            default=True)

    path_mode = path_reference_mode

    check_extension = True

    def execute(self, context):        # execute() is called by blender when running the operator.
        C = context
        obj_strs = []

        for name, v in C.scene.objects.items():
            if not v.is_visible(C.scene):
                continue
            
            if v.data.name.split('.')[0] in ['Box', 'Cylinder', 'Sphere']:
                v.rotation_mode = 'QUATERNION'
                
                type = v.data.name.split('.')[0].lower()
                pos  = list(v.location)
                quat = [v.rotation_quaternion.x, v.rotation_quaternion.y, v.rotation_quaternion.z, v.rotation_quaternion.w]
                color = (0.4, 0.4, 0.4, 1.0) if pos[2] < 0 else hsva_to_rgba(min(pos[2] / 2.0, 1.0) * 0.65, 1, 1, 1)
                color = '[{}]'.format(', '.join([str(x) for x in  color]))
                extents = list(v.dimensions)
                pose_str = '\tposition: [{}, {}, {}]\n\trotation: [{}, {}, {}, {}]'.format(*(pos + quat))
                
                obj_str = '- name: {}\n\ttype: rigid_body\n\tgeom_type: {}\n\tinitial_pose:\n{}\n\tcolor: {}\n\tmass: 0\n\textents: {}\n\tradius: {}\n\theight: {}'.format(name, type, indent(pose_str, 2), color, str(extents), extents[0]*0.5, extents[2])
                obj_strs.append(indent(obj_str, 1))
                
        out = 'world:\n{}'.format('\n'.join(obj_strs))

        f = open(self.filepath, 'w')
        f.write(out)
        f.close()
        return {'FINISHED'}            # this lets blender know the operator finished successfully.


def menu_func_import(self, context):
    self.layout.operator(ExportScene.bl_idname, text="GMM-Map (.yaml)")

def register():
    bpy.utils.register_class(ExportScene)


def unregister():
    bpy.utils.unregister_class(ExportScene)


# This allows you to run the script directly from blenders text editor
# to test the addon without having to install it.
if __name__ == "__main__":
    register()