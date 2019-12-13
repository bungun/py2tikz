from __future__ import print_function
import argparse
import yaml
import numpy as np
import subprocess

def indent(level=0):
    return int(level) * '\t'

def build_wrapper(input_file, title=''):
    wrapper = ''
    wrapper += r'\documentclass[11pt]{article}' + '\n'
    wrapper += r'\usepackage{tikz,pgfplots,filecontents,xcolor}' + 2*'\n'
    wrapper += r'\usepgfplotslibrary{groupplots}' + '\n'
    wrapper += r'\pgfplotsset{compat=1.16}' + '\n'
    wrapper += r'\usetikzlibrary{calc}' + '\n'
    wrapper += r'\usetikzlibrary{external}' + '\n'
    wrapper += r'\tikzexternalize' + '\n'
    wrapper += '\n'

    wrapper += r'\begin{document}' + '\n'

    wrapper +=r'\title{' + '{}'.format(title) + r'}' + '\n'
    wrapper +=r'\author{Bar\i\d s ungun}' + '\n'
    wrapper +=r'\date{}'
    if title:
        wrapper +=r'\maketitle' + '\n'
    wrapper += '\n'

    wrapper += r'\input{./' + input_file.replace('.tex', '') +r'}' + 2*'\n'

    wrapper += r'\end{document}'
    return wrapper

def trim_final_comma(string, cr=False, keep_cr=True):
    if string and cr and string[-2:] == ',\n':
        return string[:-2] + int(keep_cr) *'\n'
    elif string and string[-1] == ',':
        return string[:-1]
    else:
        return string

def write_header(**options):
    indent_ = indent(options.get('indent_level', 0))
    command = indent_ + r'\begin{tikzpicture}'
    command += '\n'
    return command

def write_footer(**options):
    indent_ = indent(options.get('indent_level', 0))
    command = indent_ + r'\end{tikzpicture}'
    command + '\n'
    return command

def write_plot_args(**options):
    indent_ = indent(options.get('indent_level', 2))

    tex_options = ''
    if options.get('title', None):
        tex_options += indent_ + r'title={'
        tex_options += options['title']
        tex_options += r'},'
        tex_options += '\n'
    if options.get('logx', False):
        tex_options += indent_ + 'xmode=log,\n'
        tex_options += indent_ + 'log basis x=10,\n'
    if options.get('logy', False):
        tex_options += indent_ + 'ymode=log,\n'
        tex_options += indent_ + 'log basis y=10,\n'
    for opt_name in (
            'axis_x_line', 'axis_y_line', 'xlabel', 'ylabel',
            'xtick_distance', 'ytick_distance', 'legend_pos'):
        opt = options.get(opt_name, None)
        if opt:
            tex_options += indent_ + '{}={},\n'.format(
                    opt_name.replace('_', ' '), opt)
    tex_options = trim_final_comma(tex_options, cr=True)

    return tex_options

def begin_plot(**options):
    indent_level = options.pop('indent_level', 1)
    indent_ = indent(indent_level)

    command = indent_ + r'\begin{axis}['
    command += '\n'
    command += write_plot_args(indent_level=indent_level+1, **options)
    command += indent_ + ']\n'
    return command

def end_plot(**options):
    indent_ = indent(options.get('indent_level', 1))
    command = indent_ + r'\end{axis}'
    command += '\n'
    return command

def format_series(**options):
    color = options.get('color', None)
    fillcolor = options.get('fillcolor', None)
    mark = options.get('mark', '*')
    linestyle = options.get('linestyle', None)

    fmt = '['
    if color:
        fmt += 'color={},'.format(color)
    fmt += 'mark={},'.format(mark)
    if linestyle:
        fmt += '{},'.format(linestyle)
    if fillcolor:
        fmt += r'mark options={'
        fmt += 'solid,fill={}'.format(fillcolor)
        fmt += r'}'
    fmt = trim_final_comma(fmt)
    fmt += ']'
    return fmt

def series_from_numpy(filename, **options):
    out = '\n{}\t{}\n'.format(
            options.get('xseries', 'x'),
            options.get('yseries', 'y'))

    if '.npz' in filename:
        data = np.load(filename)
        x = data[options['xseries']]
        y = data[options['yseries']]
        assert len(x) == len(y)
        for i in range(len(x)):
            out += '{}\t{}\n'.format(x[i], y[i])
        return out
    else:
        data = np.load(filename)
        for row in data:
            out += '{}\t{}\n'.format(row[0], row[1])
        return out

def write_data_series(**options):
    indent_level = options.get('indent_level', 1)
    indent_ = indent(indent_level)
    indent2_ = indent(indent_level + 1)

    xlabel = options.get('xlabel', 'x')
    ylabel = options.get('ylabel', 'y')
    xseries = options.get('xseries', 'x')
    yseries = options.get('yseries', 'y')


    command = indent_ + r'\addplot'
    command += format_series(**options)
    command += '\n' + indent2_ + 'table[x={},y={}]'.format(xseries, yseries)
    command += r'{'
    data = options['data']
    if '.np' in data:
        data = series_from_numpy(data, **options)
    command += data
    command += r'};' + '\n'
    return command

def write_computed_series(**options):
    indent_level = options.get('indent_level', 1)
    indent_ = indent(indent_level)
    indent2_ = indent(indent_level + 1)

    dom = options.get('domain', None)

    command = indent_ + r'\addplot'
    command += format_series(**options)
    command += '\n' + indent2_
    if dom:
        assert len(dom) == 2
        command += '[domain={}:{}]'.format(*dom)
    command += r'{'
    command += '{}'.format(options.get('expression', '0'))
    command += r'};' + '\n'
    return command

def write_legend(series_names=None, **options):
    if series_names is None:
        series_names = []
    indent_ = indent(options.get('indent_level', 1))
    command = indent_ + r'\legend{'
    # trim final comma
    command += sum(('{},'.format(sn) for sn in series_names))[:-1]
    command += r'}'
    return command

def write_all_series(series, **options):
    tex = ''
    for s in series:
        s.update(options)
        if s['type'] == 'raw':
            tex += write_data_series(**s)
        else:
            tex += write_computed_series(**s)
    if options.get('draw_legend', False):
        tex += write_legend(**options)
    return tex

def from_dict(d, suppress_header_footer=False, **options):
    # single plot, 1+ series
    d = dict(d)
    assert 'plot' in d
    assert 'series' in d

    options.update(d['plot'])
    series = d['series']

    tex = ''
    if not suppress_header_footer:
        tex += write_header()
        tex += begin_plot(**options)
    tex += write_all_series(series, **options)
    if not suppress_header_footer:
        tex += end_plot(**options)
        tex += write_footer(**options)
    return tex

def write_group_position(group_spec, **options):
    """
    specify group position [anchor=CARDINAL DIRECTION,at=(XCOORD, YCOORD)]
    where
        (XCOORD,YCOORD) =
            ($(REFERENCE PLOT.CARDINAL DIRECTION) + (OFFSET_X,OFFSET_Y)$)
    """
    anchor = group_spec.get('anchor', None)
    relative = group_spec.get('relative', None)
    offset = group_spec.get('offset', None)

    if not any((anchor, relative, offset)):
        return ''

    out = ''


    # build "anchor=(CARDINAL)"
    if anchor:
        out += 'anchor={},'.format(anchor)

    # build "at=(XCOORD, YCOORD)"
    position = ''
    if relative:
        position += '({})'.format(relative)
    if offset:
        if relative:
            position += ' + ({})'.format(offset)
        else:
            position += '({})'.format(offset)

    if position:
        out += r'at={($' + position + r'$)}'

    # prepend tabbing
    if out:
        out = indent(options.get('indent_level', 2)) + trim_final_comma(out)
    return out

def write_panel_labels(group_spec, **options):
    indent_level = options.get('indent_level', 1)
    indent_ = indent(indent_level)

    group_name = group_spec['group_name']
    ncols, nrows = eval(group_spec['group_layout'].replace('by', ','))

    labels = group_spec.get('labels', [])
    if not labels:
        return ''
    assert len(labels) == group_spec['group_size']
    pos = group_spec.get('label_pos', 'north west')
    offset = group_spec.get('label_offset', '0,0')

    fmt = group_spec.get('label_format', None)
    def format_label(label):
        new_label = r'{'
        new_label += '\\'
        new_label += '{} {}'.format(fmt, label)
        new_label += r'}'
        return new_label

    if fmt:
        labels = list(map(format_label, labels))

    out = ''
    for idx, label in enumerate(labels):
        # row major labeling
        col = (idx % ncols) + 1
        row = (idx / ncols) + 1

        out += indent_ + r'\node[anchor=' + pos + r'] at ($'
        out += '({} c{}r{}.{})'.format(group_name, col, row, pos)
        out += ' + ({})'.format(offset)
        out += r'$){' + label + r'};' + '\n'
    return out

def from_list(li, subplot_meta, **options):
    # multiple plots, 1+ series each
    indent_level = options.get('indent_level', 1)
    indent_ = indent(indent_level)
    indent2_ = indent(indent_level + 1)
    indent3_ = indent(indent_level + 2)

    li = iter(li)
    tex = ''

    # for group g = 1, ..., G in groups
    for group in subplot_meta:
        # open group \groupplot[GROUPSTYLE={OPTIONS}, GROUP_OPTIONS]
        tex += indent_ + r'\begin{groupplot}['
        tex += '\n'

        # GROUPSTYLE={OPTIONS}
        tex += indent2_ + 'group style={\n'
        tex += indent3_ + 'group name={},\n'.format(group['group_name'])
        tex += indent3_ + 'group size={},\n'.format(group['group_layout'])

        keys = [
            ('horizontal_sep', 'horizontal sep'),
            ('vertical_sep', 'vertical sep'),
            ('x_desc_at', 'x descriptions at'),
            ('y_desc_at', 'y descriptions at'),
        ]
        for k in keys:
            if group.get(k[0], None):
                tex += indent3_ + '{}={},\n'.format(k[1], group[k[0]])
        tex = trim_final_comma(tex, cr=True, keep_cr=False)
        tex += r'},'
        tex += '\n'

        # GROUP_OPTIONS
        tex += indent2_ + 'width={},\n'.format(group['width']) # TODO
        tex += indent2_ + 'height={},\n'.format(group['height']) # TODO
        keys = [
                ('xtick_pos', 'xtick pos'),
                ('ytick_pos', 'ytick pos'),
        ]
        for k in keys:
            if group.get(k[0], None):
                tex += indent2_ + '{}={},\n'.format(k[1], group[k[0]])
        tex += indent2_ + '{}]\n'.format(group.get('scaling', 'scale only axis'))

        # for subplots sp = 1, ..., SP in group
        counter = 0
        for sp in range(group['group_size']):
            plot_dict = next(li)

            # \nextgroupplot[OPTIONS]
            tex += indent2_ + r'\nextgroupplot'
            axis_opts = write_plot_args(
                    indent_level=indent_level + 2,
                    **plot_dict['plot'])
            if counter == 0:
                axis_opts += write_group_position(group, indent_level=indent_level + 2)
            if axis_opts:
                tex += '[' + axis_opts + ']'
            tex += '\n'

            # for series s = 1, ..., S in subplot: \addplot{DATA}
            tex += from_dict(
                    plot_dict,
                    suppress_header_footer=True,
                    indent_level=indent_level + 1,
                    **options)
            counter += 1

        # close group \end{groupplot}
        tex += indent_ + r'\end{groupplot}'
        tex += '\n'
        tex += write_panel_labels(group, indent_level=indent_level)

    return write_header(**options) + tex + write_footer(**options)

def parse_inputs(inputs, **options):
    inputs = dict(inputs)
    if 'subplot_meta' in inputs:
        sp_layout = inputs['subplot_meta']
        sp_list = inputs['subplots']
        return from_list(sp_list, sp_layout, **options)
    else:
        return from_dict(inputs)


def from_yaml(filename):
    with open(filename, 'r') as stream:
        return parse_inputs(yaml.load(stream))

def default_suffix(file, suffix):
    if not file.endswith(suffix):
        if hasattr(suffix, '__iter__'):
            suffix = suffix[0]
        return file + suffix
    else:
        return file



def test1():
    options = dict(
        xlabel= r'iterations $k$',
        ylabel = r'$\|y^k - y^\star\|$',
        logx = True,
        logy = True,
        axis_x_line = 'bottom',
        axis_y_line = 'left',
        legend_pos = 'south east'
    )

    tex = write_header()
    tex += begin_plot(**options)
    tex += write_data_series(
            xlabel='x',
            ylabel='y',
            data='data',
            color='orange',
            mark='x')
    tex += write_computed_series(
            expression=r'x^2 - x + 4',
            domain=[1,5],
            color='blue',
            mark='o',
            fillcolor='green')
    tex += end_plot()
    tex += write_footer()
    return tex

def test2():
    print(from_yaml('test.yaml'))

def test3():
    print(from_yaml('test_multi.yaml'))

def test4():
    filename = 'npz_to_plot.npz'
    x = np.linspace(1, 15)
    y = x**2 + np.random.normal(0, 0.1, size=x.size)
    np.savez(filename, iters=x, residuals=y)
    print(from_yaml('test_npz.yaml'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '-i', '--infile', type=str, default='',
            help='input YAML file')
    parser.add_argument(
            '-o', '--outfile', type=str, default='',
            help='output TEX file')
    parser.add_argument(
            '-w', '--wrapper', type=str, default='',
            help='wrapper TEX file')
    args = parser.parse_args()
    test = not (args.infile and args.outfile)

    if test:
        print(test1())
        print('\n')
        print(test2())
        print(test3())
        print(test4())
    else:
        infile = default_suffix(args.infile, ('.yaml', '.yml'))
        outfile = default_suffix(args.outfile, '.tex')
        with open(outfile, 'w') as out:
            out.write(from_yaml(infile))
            if args.wrapper:
                wrapper = default_suffix(args.wrapper, '.tex')
                with open(wrapper, 'w') as w:
                    w.write(build_wrapper(outfile))


# test1 EXPECT:
#
# \begin{tikzpicture}
#     \begin{axis}[
#         axis x line=bottom,
#         axis y line=left,
#         xmode=log,
#         ymode=log,
#         log basis x=10,
#         log basis y=10,
#         ylabel=$\|y^k-y^\star\|_2$,
#         xlabel=iteration $k$,
#         legend pos=south east
#     ]
#     \addplot[color=orange,mark=x]
#         table[x=x,y=y]{data};
#     \addplot[color=blue,mark=o,mark options={solid,fill=green}]
#         [domain=1:5]{x^2 - x + 4};
#     \end{axis}
# \end{tikzpicture}




